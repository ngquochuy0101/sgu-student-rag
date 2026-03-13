# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-07

### Added
- 🚀 Production-ready RAG system with modular architecture
- 📁 Batch PDF processing with directory scan support
- 🌍 Multilingual OCR (Vietnamese + English)
- 🔍 Semantic search using multilingual embeddings
- 💾 Persistent vector store (save/load FAISS index)
- 📊 Real-time progress tracking and statistics
- 🛡️ Comprehensive error handling and recovery
- 📝 Document separators for multi-PDF attribution
- ⚙️ Centralized configuration via `RAGConfig` class
- 🔐 Environment variable support for API keys
- 🤖 Google Gemini 2.5 Flash integration
- 📚 Detailed documentation and usage examples

### Features
- `OCRProcessor` class with single and batch processing
- `VectorStoreManager` for embedding and storage management
- `LLMManager` for Gemini API initialization
- `RAGPipeline` for end-to-end question answering
- Verbose mode with detailed logging
- Custom filtering with glob patterns
- Batch query processing
- Interactive `ask()` function for quick queries

### Technical
- Protobuf compatibility fix for TensorFlow conflicts
- Optimized chunking with RecursiveCharacterTextSplitter
- FAISS vector store with cosine similarity
- HuggingFace multilingual embeddings (paraphrase-multilingual-mpnet-base-v2)
- Type hints and docstrings throughout
- Production-grade error handling

### Documentation
- Comprehensive README.md with architecture diagrams
- Quick start guide and usage examples
- API reference documentation
- Troubleshooting guide
- Configuration guidelines
- Contributing guidelines

## [0.2.0] - 2026-03-06

### Added
- Multi-PDF processing capability
- Document merging with separators
- Enhanced error handling

### Changed
- Refactored OCR processor to OOP design
- Improved progress reporting

## [0.1.0] - 2026-03-05

### Added
- Initial project setup
- Basic OCR with Tesseract
- Single PDF processing
- Simple text extraction

---

## Planned Features

### [1.1.0] - Coming Soon
- [ ] Web UI with Gradio
- [ ] REST API endpoints
- [ ] Docker containerization
- [ ] Advanced retrieval strategies (hybrid search, re-ranking)

### [1.2.0] - Future
- [ ] Support for additional LLMs (OpenAI, Claude, Llama)
- [ ] Document versioning and updates
- [ ] Multi-tenancy support
- [ ] Cloud deployment templates

### [2.0.0] - Long-term
- [ ] Distributed processing
- [ ] Real-time document monitoring
- [ ] Advanced analytics dashboard
- [ ] Enterprise SSO integration
