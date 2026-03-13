# 🤖 RAG System - Production-Grade PDF Q&A

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Enterprise-grade Retrieval-Augmented Generation system** for Vietnamese PDF documents with OCR, vector search, and Google Gemini integration.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [Advanced Features](#-advanced-features)
- [Performance Optimization](#-performance-optimization)
- [Troubleshooting](#-troubleshooting)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

A production-ready RAG (Retrieval-Augmented Generation) system designed for processing Vietnamese educational documents. The system extracts text from PDF files using OCR, creates semantic embeddings, and answers questions using Google's Gemini LLM with accurate source attribution.

## 🧪 MLOps Workflow (v1.1.0)

The repository now includes a modular Python package and CLI designed for repeatable operations:

- **Deterministic index lifecycle** via dataset fingerprint checks
- **OCR cache** to avoid re-processing unchanged PDFs
- **Manifest tracking** in `vector_store/manifest.json`
- **Retrieval evaluation** with JSON datasets and exportable reports
- **CI unit tests** for configuration, chunking, and retrieval metrics

### Quick Commands

```bash
# 1) Install dependencies
pip install -r requirements.txt
pip install -e .

# 2) Build or refresh index (skips rebuild if dataset unchanged)
rag-sgu build-index

# 3) Ask a question
rag-sgu query --question "Muc tieu dao tao cua nganh CNTT la gi?"

# 4) Run retrieval evaluation
rag-sgu evaluate --dataset eval_data/retrieval_eval_sample.json

# 5) Inspect index manifest
rag-sgu show-manifest
```

### Use Cases

- 📚 **Educational Q&A**: Answer questions about curricula, course descriptions, and academic programs
- 📄 **Document Analysis**: Extract and synthesize information from multiple PDF documents
- 🔍 **Semantic Search**: Find relevant information across large document collections
- 🌐 **Multilingual Support**: Handles Vietnamese, English, and mixed-language documents

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │  PDF Files   │──────▶│ OCR Engine   │──────▶│ Text Cleaning│ │
│  │  (Multiple)  │      │ (Tesseract)  │      │  & Normalize │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│         │                                              │        │
│         │                                              ▼        │
│         │                                     ┌──────────────┐ │
│         │                                     │ Text Chunking│ │
│         │                                     │ (Recursive)  │ │
│         │                                     └──────────────┘ │
│         │                                              │        │
│         │                                              ▼        │
│         │                                     ┌──────────────┐ │
│         │                                     │  Embeddings  │ │
│         │                                     │ (Multilingual│ │
│         │                                     │   MPNet)     │ │
│         │                                     └──────────────┘ │
│         │                                              │        │
│         │                                              ▼        │
│         │                                     ┌──────────────┐ │
│         │                                     │Vector Store  │ │
│         │                                     │   (FAISS)    │ │
│         │                                     └──────────────┘ │
│         │                                              │        │
│         └──────────────────┬───────────────────────────┘        │
│                            │                                    │
│                            ▼                                    │
│              ┌─────────────────────────────┐                   │
│              │      User Query             │                   │
│              └─────────────────────────────┘                   │
│                            │                                    │
│                            ▼                                    │
│              ┌─────────────────────────────┐                   │
│              │  Semantic Retrieval (k=3)   │                   │
│              └─────────────────────────────┘                   │
│                            │                                    │
│                            ▼                                    │
│              ┌─────────────────────────────┐                   │
│              │  Context + Query → LLM      │                   │
│              │  (Google Gemini 2.5 Flash)  │                   │
│              └─────────────────────────────┘                   │
│                            │                                    │
│                            ▼                                    │
│              ┌─────────────────────────────┐                   │
│              │  Generated Answer +         │                   │
│              │  Source Documents           │                   │
│              └─────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **OCR Engine** | Tesseract 5.x | Extract text from PDF images |
| **PDF Processing** | PyMuPDF (fitz) | PDF manipulation and rendering |
| **Embeddings** | HuggingFace Sentence Transformers | Multilingual semantic embeddings |
| **Vector Database** | FAISS | Efficient similarity search |
| **LLM** | Google Gemini 2.5 Flash | Natural language generation |
| **Framework** | LangChain | RAG orchestration |
| **Language** | Python 3.10+ | Core implementation |

---

## ✨ Key Features

### Core Capabilities

- 🚀 **Production-Ready Architecture**: Modular, scalable, and maintainable codebase
- 📁 **Batch Processing**: Process entire directories of PDF files automatically
- 🌍 **Multilingual OCR**: Vietnamese + English language support
- 🔍 **Semantic Search**: Context-aware document retrieval using embeddings
- 💾 **Persistent Storage**: Save/load vector stores for fast reloading
- 📊 **Progress Tracking**: Real-time processing status and statistics
- ⚡ **Optimized Performance**: Configurable chunking and retrieval parameters
- 🛡️ **Error Handling**: Graceful failures with detailed error messages

### Advanced Features

- **Document Separators**: Track and attribute sources across multiple PDFs
- **Custom Filtering**: Process specific files using glob patterns
- **Verbose Mode**: Detailed logging for debugging and monitoring
- **Batch Queries**: Process multiple questions efficiently
- **Configuration Management**: Centralized settings via `RAGConfig` class
- **Environment Variables**: Secure API key management via `.env` files

---

## 📦 Prerequisites

### System Requirements

- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.10 to 3.13 (3.10/3.11 recommended)
- **RAM**: Minimum 8GB (16GB recommended for large document sets)
- **Storage**: 5GB free space (for models and vector stores)

### Required Software

1. **Tesseract OCR** (v5.0+)
   - Download: https://github.com/UB-Mannheim/tesseract/wiki
   - Language data: Vietnamese (`vie.traineddata`) and English (`eng.traineddata`)

2. **Python Packages** (see [Installation](#-installation))

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Data_processing
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install PyMuPDF pytesseract pillow
pip install langchain langchain-community langchain-text-splitters
pip install langchain-google-genai sentence-transformers faiss-cpu

# Optional: For GPU acceleration
# pip install faiss-gpu torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Tesseract OCR

**Windows:**
```powershell
# Download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki

# Install to default location: C:\Program Files\Tesseract-OCR\
# Add language packs: vie.traineddata, eng.traineddata
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-vie tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang
```

### 5. Setup Tesseract in Project

```bash
# Create OCR directory structure
mkdir -p OCR/tessdata

# Copy Tesseract executable and data files
# Windows: Copy from C:\Program Files\Tesseract-OCR\
# Linux/Mac: Tesseract is in PATH, but copy tessdata if needed
```

---

## ⚙️ Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# .env

# Google Gemini API Key (Required)
GOOGLE_API_KEY=AIzaSy...your-api-key-here

# Optional: Override default settings
# CHUNK_SIZE=1000
# CHUNK_OVERLAP=200
# RETRIEVAL_K=3
# LLM_TEMPERATURE=0.7
```

**Get your Gemini API Key:**
1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Create a new API key
4. Copy and paste into `.env` file

### 2. Configuration File

Edit settings in the notebook or create a `config.py`:

```python
class RAGConfig:
    # Paths
    BASE_DIR = Path.cwd()
    PDF_DIR = BASE_DIR / "File_PDFs"
    VECTOR_STORE_DIR = BASE_DIR / "vector_store"
    OCR_DIR = BASE_DIR / "OCR"
    
    # OCR Settings
    TESSERACT_CMD = str(OCR_DIR / "tesseract.exe")
    TESSDATA_PREFIX = str(OCR_DIR / "tessdata")
    OCR_LANGUAGES = "vie+eng"
    OCR_DPI = 2  # Rendering quality (1-4)
    
    # Text Processing
    CHUNK_SIZE = 1000          # Characters per chunk
    CHUNK_OVERLAP = 200        # Overlap between chunks
    
    # Embeddings
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_DEVICE = "cpu"   # Or "cuda" for GPU
    
    # LLM
    LLM_MODEL = "gemini-2.5-flash"
    LLM_TEMPERATURE = 0.2      # Lower = more deterministic
    LLM_MAX_TOKENS = 1024
    
    # Retrieval
    RETRIEVAL_K = 3            # Documents to retrieve per query
```

### 3. Directory Structure

```
Data_processing/
├── .env                      # API keys (DO NOT commit!)
├── .gitignore                # Ignore sensitive files
├── README.md                 # This file
├── rag_system.ipynb          # Main notebook
├── File_PDFs/                # Input PDF files
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── OCR/                      # Tesseract configuration
│   ├── tesseract.exe
│   └── tessdata/
│       ├── eng.traineddata
│       └── vie.traineddata
└── vector_store/             # FAISS index (auto-generated)
    ├── index.faiss
    └── index.pkl
```

---

## 🚀 Quick Start

### Using Jupyter Notebook

1. **Open the notebook:**
   ```bash
   jupyter notebook rag_system.ipynb
   ```

2. **Run cells sequentially:**
   - Cell 1: Environment setup (protobuf fix)
   - Cell 2: Install dependencies (first run only)
   - Cell 3-4: Import libraries and configure
   - Cell 5-7: OCR processing (process all PDFs)
   - Cell 8-9: Text chunking
   - Cell 10-12: Create vector store
   - Cell 13-14: Initialize Gemini LLM
   - Cell 15: Create RAG pipeline
   - Cell 16+: Query and test

3. **Ask questions:**
   ```python
   result = rag_pipeline.query("Mục tiêu đào tạo là gì?")
   
   # Or use convenience function
   ask("Thời gian đào tạo bao lâu?")
   ```

### Python Script Usage

Create `main.py`:

```python
from pathlib import Path
from rag_system import RAGConfig, OCRProcessor, VectorStoreManager, LLMManager, RAGPipeline

# Initialize
config = RAGConfig()
ocr = OCRProcessor(config)
vector_mgr = VectorStoreManager(config)
llm_mgr = LLMManager(config)

# Process PDFs
texts = ocr.process_directory(config.PDF_DIR)
raw_text = ocr.merge_texts(texts)

# Create vector store
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(raw_text)
vector_store = vector_mgr.create_vector_store(chunks)

# Initialize LLM and RAG
llm = llm_mgr.initialize_gemini()
rag = RAGPipeline(llm, vector_store, config)

# Query
result = rag.query("Câu hỏi của bạn?")
print(result['result'])
```

Run:
```bash
python main.py
```

---

## 💡 Usage Examples

### Example 1: Single Document Query

```python
question = "Mục tiêu đào tạo của ngành Công nghệ thông tin là gì?"

result = rag_pipeline.query(question, verbose=True)

# Output:
# ======================================================================
# ❓ QUESTION: Mục tiêu đào tạo của ngành Công nghệ thông tin là gì?
# ======================================================================
# ⏳ Processing...
# 
# ======================================================================
# 🤖 ANSWER:
# ======================================================================
# Mục tiêu đào tạo của ngành Công nghệ thông tin là...
# [Generated answer based on retrieved context]
```

### Example 2: Batch Processing

```python
questions = [
    "Thời gian đào tạo là bao lâu?",
    "Sinh viên sẽ học những môn nào?",
    "Điều kiện tốt nghiệp là gì?"
]

results = rag_pipeline.batch_query(questions)

for q, r in zip(questions, results):
    print(f"Q: {q}")
    print(f"A: {r['result']}\n")
```

### Example 3: Processing Specific PDFs

```python
# Process only files matching pattern
texts_dict = ocr_processor.process_directory(
    config.PDF_DIR,
    pattern="*CNTT*.pdf"
)

# Process with custom settings
raw_text = ocr_processor.merge_texts(
    texts_dict,
    add_separators=False  # Merge without document boundaries
)
```

### Example 4: Save and Reload Vector Store

```python
# Save after first run
vector_manager.save_vector_store()

# Later sessions: reload without reprocessing
vector_store = vector_manager.load_vector_store()
rag_pipeline = RAGPipeline(llm, vector_store, config)
```

---

## 📁 Project Structure

```
Data_processing/
│
├── rag_system.ipynb              # Main notebook (Production-ready)
├── clean_data.ipynb              # Legacy/development notebook
├── README.md                     # Documentation (this file)
├── .env                          # Environment variables
├── .gitignore                    # Git ignore patterns
│
├── File_PDFs/                    # Input documents
│   └── *.pdf                     # PDF files to process
│
├── OCR/                          # Tesseract OCR engine
│   ├── tesseract.exe             # Windows executable
│   └── tessdata/                 # Language models
│       ├── eng.traineddata
│       └── vie.traineddata
│
└── vector_store/                 # Persistent storage (generated)
    ├── index.faiss               # FAISS vector index
    └── index.pkl                 # Metadata
```

---

## 🔥 Advanced Features

### 1. Custom Prompt Engineering

Modify the prompt template in `RAGPipeline._build_prompt_template()`:

```python
template = """Bạn là chuyên gia phân tích tài liệu.

CONTEXT: {context}

QUESTION: {question}

Yêu cầu:
- Trả lời chi tiết với ví dụ cụ thể
- Trích dẫn nguồn nếu có
- Sử dụng định dạng markdown

Trả lời:"""
```

### 2. Metadata Filtering

Track document sources:

```python
from langchain.docstore.document import Document

# Create documents with metadata
docs_with_meta = [
    Document(page_content=text, metadata={"source": filename})
    for filename, text in texts_dict.items()
]

vector_store = FAISS.from_documents(docs_with_meta, embeddings)
```

### 3. Hybrid Search

Combine semantic + keyword search:

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# Create retrievers
faiss_retriever = vector_store.as_retriever()
bm25_retriever = BM25Retriever.from_texts(chunks)

# Ensemble retriever
ensemble = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)
```

### 4. Streaming Responses

For long answers:

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

---

## ⚡ Performance Optimization

### Speed Optimization

1. **Use GPU for embeddings:**
   ```python
   EMBEDDING_DEVICE = "cuda"
   ```

2. **Reduce chunk size:**
   ```python
   CHUNK_SIZE = 500  # Faster but less context
   ```

3. **Lower retrieval count:**
   ```python
   RETRIEVAL_K = 2  # Fewer documents = faster
   ```

### Quality Optimization

1. **Increase chunk overlap:**
   ```python
   CHUNK_OVERLAP = 300  # Better context continuity
   ```

2. **Higher OCR resolution:**
   ```python
   OCR_DPI = 3  # Better quality but slower
   ```

3. **Adjust temperature:**
   ```python
   LLM_TEMPERATURE = 0.1  # More deterministic answers
   ```

### Memory Optimization

1. **Process PDFs in batches:**
   ```python
   for pdf_batch in chunked(pdf_files, batch_size=5):
       process_batch(pdf_batch)
   ```

2. **Use smaller embedding model:**
   ```python
   EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   ```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Protobuf Error

**Error:**
```
TypeError: Descriptors cannot be created directly
```

**Solution:**
Ensure cell 1 runs FIRST:
```python
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
```

#### 2. Tesseract Not Found

**Error:**
```
FileNotFoundError: Tesseract not found at: ...
```

**Solution:**
- Windows: Install Tesseract and update `TESSERACT_CMD` path
- Linux/Mac: `sudo apt-get install tesseract-ocr`

#### 3. API Key Error

**Error:**
```
ValueError: Google API Key not configured!
```

**Solution:**
Create `.env` file:
```bash
GOOGLE_API_KEY=your-key-here
```

#### 4. Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Switch to CPU
EMBEDDING_DEVICE = "cpu"

# Or reduce batch size
process_directory(batch_size=1)
```

#### 5. OCR Quality Issues

**Symptoms:** Gibberish or incorrect text extraction

**Solutions:**
- Increase DPI: `OCR_DPI = 3`
- Check PDF quality (scan resolution)
- Verify language data: `vie.traineddata` and `eng.traineddata`

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Process with detailed output
texts = ocr_processor.process_directory(config.PDF_DIR, verbose=True)
```

---

## 📚 API Reference

### OCRProcessor

```python
class OCRProcessor:
    def __init__(self, config: RAGConfig)
    def process_pdf(self, pdf_path: str, verbose: bool = True) -> str
    def process_directory(self, directory: str, pattern: str = "*.pdf", verbose: bool = True) -> Dict[str, str]
    def merge_texts(self, texts_dict: Dict[str, str], add_separators: bool = True) -> str
    @staticmethod
    def clean_text(text: str) -> str
```

### VectorStoreManager

```python
class VectorStoreManager:
    def __init__(self, config: RAGConfig)
    def create_vector_store(self, texts: List[str], verbose: bool = True) -> FAISS
    def save_vector_store(self, path: Optional[str] = None)
    def load_vector_store(self, path: Optional[str] = None) -> FAISS
```

### RAGPipeline

```python
class RAGPipeline:
    def __init__(self, llm, vector_store, config: RAGConfig)
    def query(self, question: str, verbose: bool = True) -> Dict
    def batch_query(self, questions: List[str]) -> List[Dict]
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/rag-system.git
cd rag-system

# Create feature branch
git checkout -b feature/your-feature-name

# Install dev dependencies
pip install -r requirements-dev.txt
```

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings (Google style)
- Write unit tests

### Pull Request Process

1. Update documentation
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with clear description

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain**: RAG framework
- **Google Gemini**: LLM API
- **HuggingFace**: Embedding models
- **Tesseract**: OCR engine
- **FAISS**: Vector search

---

## 📧 Contact

For questions, issues, or suggestions:

- **Issues**: [GitHub Issues](https://github.com/yourusername/rag-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/rag-system/discussions)
- **Email**: your.email@example.com

---

## 🎯 Roadmap

- [ ] Web UI with Gradio/Streamlit
- [ ] Support for more LLMs (OpenAI, Claude, Llama)
- [ ] Advanced retrieval (hybrid search, re-ranking)
- [ ] Document update/versioning
- [ ] Multi-tenancy support
- [ ] REST API
- [ ] Docker deployment
- [ ] Cloud deployment guides (AWS, GCP, Azure)

---

**Made with ❤️ for the Vietnamese AI community**#   s g u - s t u d e n t - r a g  
 