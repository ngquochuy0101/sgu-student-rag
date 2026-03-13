from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RAGSettings
from .mlops import read_json, utc_now_iso, write_json


@dataclass
class EvalSample:
    question: str
    expected_keywords: list[str]
    expected_sources: list[str]


class RAGEvaluator:
    def __init__(self, settings: RAGSettings, logger):
        self.settings = settings
        self.logger = logger

    def _load_dataset(self, dataset_path: Path) -> list[EvalSample]:
        payload = read_json(dataset_path)
        if payload is None:
            raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
        if not isinstance(payload, list):
            raise ValueError("Evaluation dataset must be a JSON list")

        samples: list[EvalSample] = []
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError("Each evaluation row must be a JSON object")
            samples.append(
                EvalSample(
                    question=str(row["question"]),
                    expected_keywords=[str(item) for item in row.get("expected_keywords", [])],
                    expected_sources=[str(item) for item in row.get("expected_sources", [])],
                )
            )
        return samples

    @staticmethod
    def _keyword_hit(context: str, expected_keywords: list[str]) -> bool:
        if not expected_keywords:
            return True
        context_lower = context.lower()
        return any(keyword.lower() in context_lower for keyword in expected_keywords)

    @staticmethod
    def _source_hit(actual_sources: list[str], expected_sources: list[str]) -> bool:
        if not expected_sources:
            return True
        actual_lower = [item.lower() for item in actual_sources]
        return any(
            any(expected.lower() in source for source in actual_lower)
            for expected in expected_sources
        )

    def evaluate_retrieval(
        self,
        vector_store,
        dataset_path: Path,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        samples = self._load_dataset(dataset_path)
        effective_k = top_k or self.settings.retrieval_k

        details: list[dict[str, Any]] = []
        keyword_hits = 0
        source_hits = 0
        combined_hits = 0
        context_lengths: list[int] = []

        for sample in samples:
            docs = vector_store.similarity_search(sample.question, k=effective_k)
            context = "\n\n".join(doc.page_content for doc in docs)
            sources = [str((doc.metadata or {}).get("source", "unknown")) for doc in docs]

            keyword_ok = self._keyword_hit(context, sample.expected_keywords)
            source_ok = self._source_hit(sources, sample.expected_sources)
            combined_ok = keyword_ok and source_ok

            keyword_hits += int(keyword_ok)
            source_hits += int(source_ok)
            combined_hits += int(combined_ok)
            context_lengths.append(len(context))

            details.append(
                {
                    "question": sample.question,
                    "keyword_hit": keyword_ok,
                    "source_hit": source_ok,
                    "combined_hit": combined_ok,
                    "retrieved_sources": sources,
                    "context_chars": len(context),
                }
            )

        total = len(samples)
        if total == 0:
            raise ValueError("Evaluation dataset is empty")

        report = {
            "generated_at": utc_now_iso(),
            "dataset": str(dataset_path),
            "top_k": effective_k,
            "total_questions": total,
            "keyword_hit_rate": keyword_hits / total,
            "source_hit_rate": source_hits / total,
            "combined_hit_rate": combined_hits / total,
            "avg_context_chars": sum(context_lengths) / total,
            "details": details,
        }
        return report

    def write_report(self, report: dict[str, Any], report_name: str | None = None) -> Path:
        suffix = report_name or report["generated_at"].replace(":", "-")
        output_path = self.settings.eval_dir / f"retrieval_eval_{suffix}.json"
        write_json(output_path, report)
        self.logger.info("Saved evaluation report to %s", output_path)
        return output_path
