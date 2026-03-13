from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_settings
from .evaluation import RAGEvaluator
from .logging_utils import configure_logging
from .pipeline import RAGBuildPipeline
from .qa_service import RAGQASystem
from .vector_index import VectorIndexManager


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG SGU MLOps CLI")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Workspace directory containing File_PDFs, OCR, and vector_store.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable console logging and keep only file logs.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-index", help="Ingest PDFs and rebuild vector index")
    build_parser.add_argument("--pattern", default="*.pdf", help="PDF glob pattern")
    build_parser.add_argument("--force", action="store_true", help="Rebuild even if fingerprint is unchanged")

    query_parser = subparsers.add_parser("query", help="Run a single RAG query")
    query_parser.add_argument("--question", required=True, help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=None, help="Retriever top-k override")

    eval_parser = subparsers.add_parser("evaluate", help="Run retrieval evaluation dataset")
    eval_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSON evaluation file",
    )
    eval_parser.add_argument("--top-k", type=int, default=None, help="Retriever top-k override")

    subparsers.add_parser("show-manifest", help="Print current vector-store manifest")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    settings = load_settings(base_dir=args.base_dir)
    logger = configure_logging(settings.logs_dir, settings.run_name, verbose=not args.quiet)

    try:
        if args.command == "build-index":
            pipeline = RAGBuildPipeline(settings=settings, logger=logger)
            result = pipeline.build_index(pattern=args.pattern, force=args.force)
            print(
                json.dumps(
                    {
                        "skipped": result.skipped,
                        "documents_count": result.documents_count,
                        "chunks_count": result.chunks_count,
                        "ocr_pages": result.ocr_pages,
                        "dataset_fingerprint": result.dataset_fingerprint,
                        "manifest_path": str(result.manifest_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.command == "query":
            index_manager = VectorIndexManager(settings=settings, logger=logger)
            vector_store = index_manager.load()
            qa = RAGQASystem(settings=settings, logger=logger)
            result = qa.query(
                question=args.question,
                vector_store=vector_store,
                top_k=args.top_k,
            )
            print("ANSWER")
            print("-" * 80)
            print(result["answer"])
            print("\nSOURCES")
            print("-" * 80)
            for source in result["sources"]:
                print(f"- {source['source']} | chunk={source['chunk_id']}")
            return 0

        if args.command == "evaluate":
            index_manager = VectorIndexManager(settings=settings, logger=logger)
            vector_store = index_manager.load()
            evaluator = RAGEvaluator(settings=settings, logger=logger)
            report = evaluator.evaluate_retrieval(
                vector_store=vector_store,
                dataset_path=Path(args.dataset),
                top_k=args.top_k,
            )
            report_path = evaluator.write_report(report)
            print(
                json.dumps(
                    {
                        "dataset": report["dataset"],
                        "total_questions": report["total_questions"],
                        "keyword_hit_rate": report["keyword_hit_rate"],
                        "source_hit_rate": report["source_hit_rate"],
                        "combined_hit_rate": report["combined_hit_rate"],
                        "avg_context_chars": report["avg_context_chars"],
                        "report_path": str(report_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.command == "show-manifest":
            if not settings.manifest_path.exists():
                raise FileNotFoundError(
                    "Manifest not found. Run build-index first."
                )
            payload = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

    except Exception as error:  # noqa: BLE001
        logger.exception("Command failed: %s", error)
        print(f"ERROR: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
