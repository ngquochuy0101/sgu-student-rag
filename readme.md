# SGU Student RAG MLOps

Production-ready Retrieval-Augmented Generation (RAG) system for SGU student documents.
The project supports OCR-based PDF ingestion, FAISS indexing, Gemini-based QA, and retrieval evaluation.

## Key Capabilities

- Hybrid ingestion: direct PDF text extraction with OCR fallback
- Deterministic index lifecycle using dataset fingerprints
- OCR cache to reduce repeated processing
- Source-traceable chunks (`source`, `chunk_id`, `source_hash`)
- CLI workflow for build, query, evaluate, and manifest inspection
- Unit tests and CI for core MLOps behaviors

## Architecture

1. Ingestion: scan PDFs in `File_PDFs/`
2. Extraction: direct text first, OCR fallback for low-text pages
3. Chunking: recursive splitting with overlap
4. Embedding: sentence-transformers multilingual model
5. Indexing: FAISS vector store in `vector_store/`
6. Retrieval + QA: retrieve relevant chunks and generate answer
7. Evaluation: run retrieval metrics from JSON evaluation datasets

## Requirements

- OS: Windows, Linux, or macOS
- Python: 3.10 to 3.13 (3.10/3.11 recommended)
- Tesseract OCR installed and available via `OCR/` folder or system path

## Installation

```bash
git clone <your-repo-url>
cd sgu-student-rag-mlops

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Configuration

1. Copy environment template:

```bash
cp .env.example .env
```

2. Set required key in `.env`:

```env
GOOGLE_API_KEY=your_key_here
```

3. Optional overrides:

```env
RAG_RUN_NAME=dev-local
RAG_PDF_DIR=File_PDFs
RAG_VECTOR_STORE_DIR=vector_store
RAG_OCR_DIR=OCR
RAG_ARTIFACTS_DIR=artifacts
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4
OCR_LANGUAGES=vie+eng
OCR_DPI=2
OCR_PSM=3
OCR_OEM=1
OCR_MIN_TEXT_CHARS=60
OCR_CACHE_ENABLED=true
```

## CLI Usage

### 1) Build or refresh index

```bash
rag-sgu build-index
```

- Rebuild is skipped automatically when dataset fingerprint is unchanged.
- Use force rebuild:

```bash
rag-sgu build-index --force
```

### 2) Query the system

```bash
rag-sgu query --question "Muc tieu dao tao cua nganh CNTT la gi?"
```

### 3) Evaluate retrieval quality

```bash
rag-sgu evaluate --dataset eval_data/retrieval_eval_sample.json
```

### 4) Inspect manifest

```bash
rag-sgu show-manifest
```

Manifest is stored at `vector_store/manifest.json`.

## Evaluation Dataset Format

Example row in evaluation JSON:

```json
{
  "question": "Huong dan dang ky mon hoc hoc ky 252 nhu the nao?",
  "expected_keywords": ["dang ky", "mon hoc", "hoc ky"],
  "expected_sources": ["HK252_HD_DangKyMonHoc"]
}
```

## Project Structure

```text
.
├── src/
│   └── rag_sgu/
│       ├── cli.py
│       ├── config.py
│       ├── document_ingestion.py
│       ├── text_chunking.py
│       ├── vector_index.py
│       ├── qa_service.py
│       ├── evaluation.py
│       ├── pipeline.py
│       └── mlops.py
├── tests/
├── eval_data/
├── docs/
├── File_PDFs/
├── OCR/
├── vector_store/
├── requirements.txt
└── pyproject.toml
```

## Notebooks

- `rag_system.ipynb`: legacy exploration notebook
- `test_api.ipynb`: API/model listing test notebook

CLI and `src/rag_sgu` modules are the recommended production path.

## Run Tests

```bash
pytest
```

## CI

GitHub Actions workflow is defined in `.github/workflows/ci.yml`.

## Troubleshooting

### `build-index` fails with OCR dependency errors

Install full dependencies:

```bash
pip install -r requirements.txt
```

### `show-manifest` says manifest not found

Run index build first:

```bash
rag-sgu build-index
```

### Python 3.14 incompatibility

Some pinned dependencies are not stable on 3.14 yet. Use Python 3.10 to 3.13.

## License

MIT License. See `LICENSE`.