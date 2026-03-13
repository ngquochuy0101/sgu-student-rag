---
description: "Use when editing RAG ingestion, vector indexing, retrieval QA, and OCR flows for this project."
applyTo: ["**/*.py", "**/*.ipynb", "**/*.md"]
---

# SGU Student RAG Instructions

## Goal
Maintain a reliable student-facing RAG system with reproducible behavior and low hallucination risk.

## Mandatory Rules
1. Prefer context-grounded answers from retrieved documents only.
2. If evidence is missing, explicitly state that information is not found in sources.
3. Keep Vietnamese support stable for OCR and answering.
4. Preserve metadata traceability (`source`, `chunk_id`, and retrieval diagnostics).
5. Keep ingestion, indexing, and QA code modular and testable.

## Engineering Constraints
- Use environment-based configuration for credentials and model/runtime settings.
- Add/update unit tests for any non-trivial logic change.
- Keep CLI workflows reproducible (`build-index`, `query`, `evaluate`).
- Log operational events that affect retrieval quality (OCR fallback, empty docs, index refresh).
