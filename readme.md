# 📚 RAG SGU — Hệ thống Hỏi đáp Tài liệu SGU

Hệ thống **Retrieval-Augmented Generation (RAG)** thông minh dành riêng cho việc tham khảo tài liệu của Đại học Sài Gòn (SGU). Dự án sử dụng kết hợp **LangChain**, cơ sở dữ liệu vector **FAISS**, **Sentence-Transformers** và sức mạnh của **Google Gemini API** để truy xuất văn bản tự nhiên.

Hệ thống cung cấp hai trải nghiệm:
- **Jupyter Notebook**: Phục vụ việc nghiên cứu, thử nghiệm và debug riêng lẻ cho lập trình viên.
- **Web App**: Giao diện Streamlit dễ sử dụng cho người dùng cuối (Sinh viên/Giảng viên), bao gồm khả năng chat, quản lý user và nhật ký truy vấn theo phân quyền.

---

## ✨ Tính năng Nổi bật
- 🔍 **Truy xuất tài liệu PDF chính xác** thông qua nền tảng FAISS Vector Search.
- 🤖 **Sinh câu trả lời thông minh** bằng Google Gemini LLM được neo vào ngữ cảnh của tài liệu được trích xuất (Tránh "ảo giác" AI).
- 🌐 **Giao diện Web mượt mà** với Streamlit, hiển thị nguồn tham khảo (Tên file & preview đoạn text) dưới dạng hộp thoại mở rộng.
- 👤 **Quản lý phân quyền đăng nhập** bằng MSSV & Ngày sinh, được lưu trữ an toàn bằng mật khẩu mã hoá cục bộ.
- 📖 **Hỗ trợ tiền xử lý OCR** dành cho các bản scan PDF khó đọc tiếng Việt bằng thư viện Tesseract nội bộ.

---

## 💻 Yêu cầu Hệ thống (Prerequisites)
Trước khi tiến hành thao tác cài đặt, hãy đảm bảo máy tính bạn có:
1. **Hệ điều hành**: Khuyến nghị Windows 10/11. (Có thể chạy trên macOS/Linux với một số tuỳ chỉnh đường dẫn phụ).
2. **Python**: Cài đặt Python từ phiên bản **`3.10` đến `3.13`**. (Vui lòng tick chọn ký hiệu *"Add python.exe to PATH"* ở trình cài đặt).
3. **Tesseract OCR** *(Rất quan trọng nếu file PDF của bạn là file scan hình ảnh)*:
   - Truy cập vào trang [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) để tải bộ cài exe.
   - Cài đặt vào thư mục mặc định: `C:\Program Files\Tesseract-OCR`.
4. **Google Gemini API Key**:
   - Truy cập [Google AI Studio](https://aistudio.google.com/) bằng tài khoản Google bất kỳ để khởi tạo miễn phí một mã API Key.

---

## 🚀 Hướng dẫn Cài đặt & Setup Dự án

### Bước 1: Tạo mô trường ảo (Virtual Environment)
Một môi trường ảo (Virtual Env) giúp bạn cài các thư viện của dự án này mà không làm ảnh hưởng tính ổn định với các dự án Python khác trên máy.
Mở Terminal (Command Prompt hoặc PowerShell) tại thư mục nguồn dự án (`sgu-student-rag`) và chạy tuần tự cấu trúc:
```bash
# Tạo môi trường ảo có thư mục mang tên .venv
python -m venv .venv
```

**Kích hoạt môi trường ảo vừa tạo:**
- Trên **Command Prompt (CMD)**: `.\.venv\Scripts\activate.bat`
- Trên **PowerShell**: `.\.venv\Scripts\Activate.ps1`
- Trên macOS/Linux: `source .venv/bin/activate`

*(**Lưu ý**: Sau khi thực hiện lệnh kích hoạt thành công, bạn sẽ thấy tiếp đầu ngữ cờ `(.venv)` xuất hiện ở đầu dấu nhắc lệnh gõ command).*

### Bước 2: Cài đặt thư viện phụ thuộc
Sau khi đã chắc chắn bạn đang đứng trong môi trường ảo, tiến hành tải bộ các thư viện cần dùng với cú pháp:
```bash
pip install -r requirements.txt
```

### Bước 3: Cấu hình tham số môi trường
Dự án lưu các thông số tuỳ biến và API Key ở dạng file biến riêng tư tên là `.env`. Trước hết, hãy nhân bản file mẫu `example` cho sẵn:
```bash
# Lệnh nếu bạn xài Windows Command Prompt
copy .env.example .env

# Lệnh nếu bạn xài PowerShell
Copy-Item .env.example .env
```
Mở file `.env` vừa được sinh ra bằng ứng dụng Notepad hoặc VS Code. Tìm từ khoá `GOOGLE_API_KEY=YOUR_API_KEY_HERE` và thế bằng mã Google API Key bạn lấy được vào dòng lệnh.
*(Ghi chú: Tại đây, bạn hoàn toàn có thể tự điều chỉnh các tham số cấu hình nhỏ như `CHUNK_SIZE` hay thư mục `RAG_PDF_DIR` tuỳ ý thích sau này)*.

### Bước 4: Nạp Tài liệu PDF vào dự án (Quá trình tải Data)
Mọi file PDF được dùng làm CSDL nguồn sẽ cần phải được đọc. Mặc định nó sẽ đọc toàn bộ PDF qua một thư mục có tên `File_PDFs_OCR`.
- **Nếu là File PDF chữ gõ văn bản thường**: Chỉ cần tạo thư mục mang tên `File_PDFs_OCR` ở thư mục gốc của project này và bỏ các tờ bài giảng/công văn (file .pdf) của trường vào nó.
- **Nếu là File PDF của bạn chụp Scan mờ (không bôi đen text được)**: Hãy xử lý ảnh (bỏ qua script chuẩn bị):
  ```bash
  python ocr_pdf.py --input <Thư_mục_chứa_PDF_hình> --output File_PDFs_OCR --dpi 400
  ```
  *(Các file in sẽ được Tesseract OCR quét và cấy đè layer văn bản text chìm).*

---

## ▶️ Khởi chạy Ứng dụng Web Chatbot
Sau khi thực hiện cấu hình trót lọt, để mở ứng dụng hỏi đáp trực quan Web:

- **Cách 1 - Khởi chạy cực nhanh**: Nhấn đúp chuột (Double-Click) vào file `run_web.cmd`. Script này đã được tối ưu để tự tìm môi trường và bung website trực diện.
- **Cách 2 - Qua terminal PowerShell (Nếu bị Admin block File Exec)**:
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\run_web.ps1
  ```
- **Cách 3 - Trực tiếp từ file (Dành cho Lập trình viên)**:
  ```bash
  python -m streamlit run streamlit_app.py
  ```

> 🌐 Giao diện Streamlit của dự án sẽ tự động render trên trình duyệt tại địa chỉ: **http://localhost:8501**
> **Lưu ý:** Nếu là lần đầu chạy và Web hỏi, hãy ấn nút `"Nạp lại RAG index"` ở bên góc trái giao diện để engine lập chỉ mục ban đầu!

### 🔐 Thông tin Đăng nhập vào Web
Nhằm khoá giới hạn truy vấn tới đối tượng xác định, bạn cần đăng nhập.
- **Tài khoản Administrator hệ thống mặc định**: 
  - MSSV: **`admin`**
  - Ngày sinh: **`01/01/2000`**
  *(Có thể trỏ lại mật khẩu tài khoản Admin gốc nếu muốn thông qua file `.env`)*.
- Thông qua cửa sổ **"Quản lý người dùng"** từ tài khoản Admin, bạn có thể tự thân cấp phát MSSV riêng lẽ cho người khác. Họ có thể đăng nhập bằng **Mã sinh viên** kèm ngày cấp dưới dạng (VD: 21/04/2004).

---

## 📓 Hướng Dẩn cho Môi trường Nghiên Cứu (Jupyter)
Dành cho Data Scientist không mong muốn chạy Web UI. Dự án có kèm tệp `rag_system.ipynb` phục vụ pipeline phân tích rời rạc:
1. Mở file `rag_system.ipynb` trên giao diện VS Code, DataSpell hoặc JupyterLab.
2. Thiết lập kernel của IDE Python trỏ đúng về vị trí interpreter ảo bạn cài lúc nãy `(.*sgu-student-rag/.venv/Scripts/python.exe)`.
3. Chạy từng Block theo quy trình từ Khởi tạo `[Config]` ➜ Build Embed `[FAISS Index]` ➜ Viết `[Prompt Query]` để hiểu cơ chế làm việc cơ bản.

---

## 📂 Tổ chức Cấu trúc Code Tổng Quan
```text
.
├── .env.example            # Mẫu tham số môi trường
├── readme.md               # Tài liệu bạn đang đọc (Hướng dẫn)
├── requirements.txt        # Tập lưu trữ thư viện Python
├── run_web.cmd             # Script Launch Web siêu tốc
├── run_web.ps1             # Cú pháp chạy Web PowerShell mở rộng
├── ocr_pdf.py              # Xử lý Engine Tesseract nhúng OCR cho file PDF ảnh
├── rag_system.ipynb        # Thực thi mô hình hóa Test RAG bằng Notebook
├── streamlit_app.py        # Framework GUI App Web chính hãng
├── src/                    # Kiến trúc lõi (Back-End RAG)
│   └── rag_core/           # (chứa script: config, nạp pdf, chia lô text, qa-bot)
├── File_PDFs_OCR/          # [Thư mục user] - Chứa data PDF gốc (Cần tạo)
├── vector_store/           # [Trình tạo DB] Lữu trữ Vector chỉ mục của FAISS
└── artifacts/              # [Trình tạo System] Logs File & User Database nhỏ gọn
```
