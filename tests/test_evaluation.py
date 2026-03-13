import json
import logging
from pathlib import Path
from types import SimpleNamespace

from rag_sgu.config import load_settings
from rag_sgu.evaluation import RAGEvaluator


class FakeVectorStore:
    def __init__(self, lookup):
        self.lookup = lookup

    def similarity_search(self, question: str, k: int):
        rows = self.lookup[question][:k]
        return [
            SimpleNamespace(page_content=item["text"], metadata={"source": item["source"]})
            for item in rows
        ]


def test_evaluate_retrieval_reports_hits(tmp_path):
    dataset_path = tmp_path / "eval.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question": "Q1",
                    "expected_keywords": ["dao tao"],
                    "expected_sources": ["BanMOTa"],
                },
                {
                    "question": "Q2",
                    "expected_keywords": ["dang ky"],
                    "expected_sources": ["HK252"],
                },
            ]
        ),
        encoding="utf-8",
    )

    lookup = {
        "Q1": [{"text": "Thong tin dao tao nganh CNTT", "source": "BanMOTa_CNTT_2020-2024.pdf"}],
        "Q2": [{"text": "Huong dan dang ky mon hoc", "source": "HK252_HD_DangKyMonHoc.pdf"}],
    }
    vector_store = FakeVectorStore(lookup)

    settings = load_settings(base_dir=tmp_path)
    evaluator = RAGEvaluator(settings=settings, logger=logging.getLogger("test"))

    report = evaluator.evaluate_retrieval(
        vector_store=vector_store,
        dataset_path=Path(dataset_path),
        top_k=1,
    )

    assert report["total_questions"] == 2
    assert report["keyword_hit_rate"] == 1.0
    assert report["source_hit_rate"] == 1.0
    assert report["combined_hit_rate"] == 1.0

    report_path = evaluator.write_report(report, report_name="unit")
    assert report_path.exists()
