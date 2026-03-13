# RAG MLOps Runbook

This runbook defines the standard operation cycle for the SGU RAG system.

## 1. Preparation

1. Install dependencies.
2. Configure `.env` with `GOOGLE_API_KEY` and optional overrides.
3. Verify PDF inputs exist in `File_PDFs/`.

## 2. Build or Refresh Index

```bash
rag-sgu build-index
```

Notes:
- The command compares PDF metadata fingerprint with `vector_store/manifest.json`.
- If unchanged, rebuild is skipped unless `--force` is passed.

## 3. Query Validation

```bash
rag-sgu query --question "Muc tieu dao tao cua nganh CNTT la gi?"
```

Expected output:
- Answer text
- Source chunk list with document names

## 4. Retrieval Evaluation

```bash
rag-sgu evaluate --dataset eval_data/retrieval_eval_sample.json
```

Generated artifacts:
- JSON report under `artifacts/evaluations/`
- Metrics: keyword hit, source hit, combined hit, avg context chars

## 5. Operational Logs

- Logs are written to `artifacts/logs/rag_<run_name>.log`.
- Override run tag using `RAG_RUN_NAME` in `.env`.

## 6. CI Policy

- Every push or PR triggers tests in `.github/workflows/ci.yml`.
- Unit tests validate config loading, chunk metadata, fingerprint stability, and retrieval scoring.

## 7. Suggested Weekly Maintenance

1. Add 5 to 10 new evaluation questions to dataset.
2. Run `rag-sgu evaluate` and track trends in combined hit rate.
3. Rebuild index if PDF set changed.
4. Review logs for OCR failures and add corrective OCR settings when needed.
