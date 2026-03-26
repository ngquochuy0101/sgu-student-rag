# 📚 RAG SGU — Hệ thống Hỏi đáp Tài liệu SGU

Hệ thống **Retrieval-Augmented Generation (RAG)** cho tài liệu Đại học Sài Gòn, sử dụng **LangChain**, **FAISS**, **Sentence-Transformers** và **Gemini API**.

> Hỗ trợ 2 cách sử dụng: **Notebook** (nghiên cứu/debug) và **Web App** (Streamlit — đăng nhập, chat, quản lý user).

---

## ✨ Tính năng chính

- 🔍 Truy xuất tài liệu PDF với FAISS vector search
- 🤖 Trả lời câu hỏi bằng Gemini LLM kết hợp ngữ cảnh truy xuất
- 🌐 Giao diện web Streamlit với hệ thống đăng nhập (MSSV + ngày sinh)
- 👤 Quản lý người dùng (admin/user) và nhật ký chat
- 📄 Hiển thị nguồn tham khảo (file + trang) cho mỗi câu trả lời

---

## 📂 Cấu trúc dự án

```
.
├── .env.example            # Mẫu biến môi trường
├── readme.md
├── requirements.txt
├── run_web.cmd              # Khởi chạy web app (CMD)
├── run_web.ps1              # Khởi chạy web app (PowerShell)
├── ocr_pdf.py               # OCR PDF scan -> PDF có text layer
├── rag_system.ipynb         # Notebook chạy pipeline RAG
├── streamlit_app.py         # Web app chính
├── src/
│   └── rag_core/
│       ├── __init__.py
│       ├── config.py        # Cấu hình tập trung (dataclass)
│       ├── environment.py   # Setup runtime Windows
│       ├── ingestion.py     # Ingest PDF (PyPDFLoader)
│       ├── chunking.py      # Chia nhỏ tài liệu
│       ├── vector_store.py  # Build/Load/Save FAISS index
│       ├── qa_service.py    # QA service (Gemini + retrieval)
│       └── pipeline.py      # Orchestration cho notebook
├── File_PDFs/               # PDF gốc
├── File_PDFs_OCR/           # (tùy chọn) PDF đã OCR, dùng làm input mặc định
├── vector_store/            # FAISS index (tự động tạo)
├── artifacts/               # DB, cache, logs runtime
└── eval_data/               # Dữ liệu đánh giá
```

---

## 🚀 Cài đặt

### 1. Tạo môi trường ảo và cài dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường

Sao chép `.env.example` thành `.env` và điền API key:

```cmd
copy .env.example .env
```

```powershell
Copy-Item .env.example .env
```

```env
GOOGLE_API_KEY=your_api_key_here
```

### 3. Chuẩn bị dữ liệu PDF

- Mặc định pipeline đọc từ `File_PDFs_OCR/` (biến `RAG_PDF_DIR`).
- Nếu chưa có thư mục này, bạn có thể tạo và đặt PDF đã có text layer vào đó.
- Nếu PDF là scan/image-only, chạy OCR trước:

```bash
python ocr_pdf.py --input File_PDFs --output File_PDFs_OCR --dpi 400
```

---

## ▶️ Chạy ứng dụng

### Cách 1 — Double-click (đơn giản nhất)

```
run_web.cmd
```

### Cách 2 — PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\run_web.ps1
```

### Cách 3 — Chạy trực tiếp

```bash
python -m streamlit run streamlit_app.py
```

> Web app mặc định chạy tại **http://localhost:8501**

---

## 📓 Chạy Notebook

Mở `rag_system.ipynb` và chạy theo thứ tự cell:

1. Setup path + environment
2. Khởi tạo `RAGConfig` và `RAGPipeline`
3. Build FAISS index từ PDF
4. Query câu hỏi và xem kết quả

---

## ⚙️ Cấu hình

Các tham số chính (đặt trong `.env` hoặc dùng giá trị mặc định):

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `RAG_PDF_DIR` | `File_PDFs_OCR` | Thư mục chứa PDF |
| `CHUNK_SIZE` | `1600` | Kích thước chunk |
| `CHUNK_OVERLAP` | `200` | Overlap giữa các chunk |
| `RETRIEVAL_K` | `4` | Số document truy xuất |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Model embedding |
| `LLM_MODEL` | `gemini-2.5-flash` | Model LLM |
| `LLM_TEMPERATURE` | `0.5` | Nhiệt độ sinh câu trả lời |

---

## 🔐 Đăng nhập

- Đăng nhập bằng **MSSV** + **Ngày sinh** (dd/mm/yyyy)
- Tài khoản admin mặc định: `admin` / `01/01/2000`
- Có thể thay đổi qua biến `RAG_ADMIN_MSSV` và `RAG_ADMIN_BIRTH_DATE` trong `.env`

---

## 🛠️ Tech Stack

- **LangChain** — Orchestration RAG pipeline
- **FAISS** — Vector similarity search
- **Sentence-Transformers** — Multilingual embeddings
- **Google Gemini** — LLM sinh câu trả lời
- **Streamlit** — Web UI
- **SQLite** — Lưu trữ user và chat logs
