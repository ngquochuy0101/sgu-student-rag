# RAG SGU - PyPDFLoader Minimal

He thong RAG toi gian cho tai lieu SGU voi 2 entrypoint:
- Notebook: `rag_system.ipynb`
- Web app: `streamlit_app.py`

Pipeline core dung `DirectoryLoader + PyPDFLoader` de ingest PDF co text-layer.

## Muc tieu hien tai

- Dung chung mot code core cho notebook va web
- Cau hinh mac dinh dong bo de web tra loi sat voi notebook
- Hien thi nguon tham khao ro rang (file + trang)
- Fallback out-of-scope/not-found theo cau chuan:
	- `Toi khong tim thay thong tin nay trong tai lieu`

## Kien truc

Code core nam trong `src/rag_core`:
- `environment.py`: runtime env setup cho Windows
- `config.py`: dataclass cau hinh tap trung
- `ingestion.py`: ingest PDF bang `DirectoryLoader + PyPDFLoader`
- `chunking.py`: chunk tai lieu bang `RecursiveCharacterTextSplitter`
- `vector_store.py`: build/load/save FAISS
- `qa_service.py`: QA service (Gemini + retrieval + source labels)
- `pipeline.py`: orchestration class cho notebook/script

`streamlit_app.py` la UI (dang nhap, chat, quan ly user, chat logs) va goi `RAGService` tu core package.

## Cau truc thu muc

```text
.
├── .env.example
├── readme.md
├── requirements.txt
├── run_web.cmd
├── run_web.ps1
├── rag_system.ipynb
├── test_api.ipynb
├── streamlit_app.py
├── src/
│   └── rag_core/
│       ├── __init__.py
│       ├── chunking.py
│       ├── config.py
│       ├── environment.py
│       ├── ingestion.py
│       ├── pipeline.py
│       ├── qa_service.py
│       └── vector_store.py
├── File_PDFs/
├── File_PDFs_OCR/
├── vector_store/
├── artifacts/
│   ├── cache/
│   ├── evaluations/
│   └── logs/
└── eval_data/
```

Luu y:
- `File_PDFs` co the la scan/image-only.
- `File_PDFs_OCR` dung de dong bo ket qua voi notebook.

## Cai dat

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Tao `.env` tu `.env.example` va set API key:

```env
GOOGLE_API_KEY=your_key_here
```

## Bo tham so mac dinh dong bo notebook

Cac gia tri duoi day duoc dung trong code va `.env.example`:
- `RAG_PDF_DIR=File_PDFs_OCR`
- `RAG_PDF_GLOB=*.pdf`
- `RAG_VECTOR_STORE_DIR=vector_store`
- `RAG_DEMO_DB_PATH=artifacts/rag_demo.db`
- `CHUNK_SIZE=1600`
- `CHUNK_OVERLAP=200`
- `RETRIEVAL_K=4`
- `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- `EMBEDDING_DEVICE=cpu`
- `LLM_MODEL=gemini-2.5-flash`
- `LLM_TEMPERATURE=0.5`
- `LLM_MAX_TOKENS=1024`
- `LLM_API_TRANSPORT=rest`

## Chay notebook

Mo `rag_system.ipynb` va chay theo thu tu cell:
1. Setup path + env
2. Khoi tao `RAGConfig` va `RAGPipeline`
3. Build index tu PDF (neu co text-layer)
4. Neu build that bai, load index da ton tai
5. Query in-scope
6. Query out-of-scope de check fallback

Notebook hien them top retrieved passages de debug retrieval.

## Chay web app

Lua chon 1:

```bash
run_web.cmd
```

Lua chon 2:

```bash
powershell -ExecutionPolicy Bypass -File .\run_web.ps1
```

Lua chon 3:

```bash
streamlit run streamlit_app.py
```

## Tham so giao dien tren web

Trang chat co cac tham so UI trong sidebar:
- `Top-k truy xuat`
- `Hien thi top doan truy xuat`
- `So doan hien thi`
- `So ky tu moi doan`

Mac dinh duoc dat de gan voi notebook (top-k theo config, passage preview 5 doan, 180 ky tu/doan).

Nut `Nap lai RAG index` tren sidebar se clear cache service va nap lai index/config moi nhat.

## Hanh vi voi PDF scan/image-only

Voi pipeline hien tai (khong OCR), neu PDF khong co text-layer:
- Ingestion bao ro file nghi van scan/image-only
- Khong crash ung dung
- Co the fallback sang FAISS index da build truoc do

## Verification checklist

- Notebook build/load index thanh cong
- Streamlit login/chat/logs hoat dong
- Query in-scope tra loi day du + co sources
- Query out-of-scope tra fallback dung cau chuan
- UI va config web hien dung thong so dang dung
