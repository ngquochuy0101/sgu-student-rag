# RAG SGU - PyPDFLoader Minimal

He thong RAG toi gian cho tai lieu SGU voi 2 entrypoint:
- Notebook: `rag_system.ipynb`
- Web app: `streamlit_app.py`

Pipeline chi dung `DirectoryLoader + PyPDFLoader` de ingest PDF co text-layer.

## Scope hien tai

- Khong OCR
- Khong CLI workflow
- Retrieval co ban: `similarity_search` voi `k` co dinh
- Tra loi kem nguon tham khao da dedupe va page label
- Fallback not-found: `Toi khong tim thay thong tin nay trong tai lieu`

## Kien truc

Code core dung chung nam trong `src/rag_core`:

- `environment.py`: bien moi truong runtime cho Windows
- `config.py`: dataclass cau hinh tap trung
- `ingestion.py`: ingest PDF bang `DirectoryLoader + PyPDFLoader`
- `chunking.py`: chunk tai lieu bang `RecursiveCharacterTextSplitter`
- `vector_store.py`: build/load/save FAISS
- `qa_service.py`: QA service (Gemini + retrieval + citation labels)
- `pipeline.py`: class orchestration cho notebook/script

`streamlit_app.py` giu UI dang nhap, quan ly user, va chat logs; phan RAG da import tu core package.

## Cau truc thu muc

```text
.
├── src/
│   └── rag_core/
├── rag_system.ipynb
├── streamlit_app.py
├── requirements.txt
├── File_PDFs/
├── vector_store/
└── artifacts/
```

## Cai dat

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Tao file `.env` tu `.env.example` va set API key:

```env
GOOGLE_API_KEY=your_key_here
```

Luu y voi du lieu mau trong repo:
- `File_PDFs` hien la scan/image-only, PyPDFLoader se can text-layer de rebuild index moi.
- Co the set `RAG_PDF_DIR=File_PDFs_OCR` (PDF da OCR san) hoac su dung index co san trong `vector_store`.

## Bien moi truong chinh

- `RAG_PDF_DIR` (mac dinh: `File_PDFs`)
- `RAG_PDF_GLOB` (mac dinh: `*.pdf`)
- `RAG_VECTOR_STORE_DIR` (mac dinh: `vector_store`)
- `RAG_DEMO_DB_PATH` (mac dinh: `artifacts/rag_demo.db`)
- `CHUNK_SIZE` (mac dinh: `1000`)
- `CHUNK_OVERLAP` (mac dinh: `200`)
- `RETRIEVAL_K` (mac dinh: `6`)
- `EMBEDDING_MODEL` (mac dinh: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`)
- `EMBEDDING_DEVICE` (mac dinh: `cpu`)
- `LLM_MODEL` (mac dinh: `gemini-2.5-flash`)
- `LLM_TEMPERATURE` (mac dinh: `0.2`)
- `LLM_MAX_TOKENS` (mac dinh: `1024`)
- `LLM_API_TRANSPORT` (mac dinh: `rest`)

## Chay notebook

Mo `rag_system.ipynb` va chay theo thu tu cell:

1. Setup path + env
2. Khoi tao `RAGConfig` va `RAGPipeline`
3. Build index tu PDF (neu co text-layer)
4. Neu build that bai, load index da ton tai
5. Query demo trong tai lieu
6. Query out-of-scope de kiem tra fallback

## Chay web app

```bash
streamlit run streamlit_app.py
```

Web app yeu cau FAISS index da ton tai trong `vector_store/`.

## Hanh vi voi PDF scan/image-only

Voi pipeline hien tai (khong OCR), neu PDF khong co text-layer:

- Ingestion se bao ro file nghi van scan/image-only
- Khong crash ung dung
- Co the fallback sang load FAISS index da build truoc do

## Verification de xuat

- Notebook build/load index thanh cong, query tra answer + sources
- Streamlit login/chat/logs hoat dong
- Query ngoai tai lieu tra fallback dung cau chuan
- Khong con import/goi OCR libraries trong code path chinh
