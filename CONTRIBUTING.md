# Contributing to RAG System

First off, thank you for considering contributing to RAG System! 🎉

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Guidelines](#development-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- ✅ Be respectful and inclusive
- ✅ Accept constructive criticism
- ✅ Focus on what is best for the community
- ✅ Show empathy towards other community members
- ❌ No harassment, trolling, or derogatory comments
- ❌ No personal or political attacks

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Tesseract OCR
- Basic understanding of RAG systems

### Development Setup

1. **Fork the repository**
   ```bash
   # Click "Fork" button on GitHub
   git clone https://github.com/YOUR_USERNAME/rag-system.git
   cd rag-system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Verify setup**
   ```bash
   python -c "import langchain, sentence_transformers; print('Setup OK!')"
   ```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When you create a bug report, include as many details as possible:

**Bug Report Template:**

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment:**
 - OS: [e.g. Windows 11, Ubuntu 22.04]
 - Python version: [e.g. 3.10.5]
 - Package versions: [output of `pip list`]

**Additional context**
Add any other context about the problem here.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Provide the following information:

- Use a clear and descriptive title
- Provide a step-by-step description of the suggested enhancement
- Provide specific examples to demonstrate the steps
- Describe the current behavior and expected behavior
- Explain why this enhancement would be useful

### Pull Requests

Good pull requests are always welcome! Here's a quick guide:

1. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes**
   - Write clear, commented code
   - Follow the coding style guidelines
   - Add tests if applicable
   - Update documentation

3. **Commit your changes**
   ```bash
   git commit -m "feat: add amazing feature"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

5. **Open a Pull Request**
   - Use the PR template
   - Link any related issues
   - Request reviews

## Development Guidelines

### Code Style

We follow PEP 8 with some modifications:

```python
# Good
class RAGProcessor:
    """Process documents for RAG system.
    
    Args:
        config: Configuration object
        verbose: Enable verbose logging
    """
    
    def __init__(self, config: RAGConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
    
    def process(self, document: str) -> List[str]:
        """Process a single document.
        
        Args:
            document: Input document text
            
        Returns:
            List of processed chunks
            
        Raises:
            ValueError: If document is empty
        """
        if not document:
            raise ValueError("Document cannot be empty")
        
        return self._split_document(document)
```

**Key Points:**

- ✅ Use type hints
- ✅ Add docstrings (Google style)
- ✅ 4 spaces for indentation
- ✅ Max line length: 100 characters
- ✅ Meaningful variable names
- ✅ Comments for complex logic

### Testing

Add tests for new features:

```python
# tests/test_ocr.py
import pytest
from rag_system import OCRProcessor

def test_ocr_processor_init():
    """Test OCR processor initialization"""
    processor = OCRProcessor(config)
    assert processor.config is not None

def test_process_pdf_success():
    """Test successful PDF processing"""
    processor = OCRProcessor(config)
    result = processor.process_pdf("test.pdf")
    assert len(result) > 0

def test_process_pdf_not_found():
    """Test PDF not found error"""
    processor = OCRProcessor(config)
    with pytest.raises(FileNotFoundError):
        processor.process_pdf("nonexistent.pdf")
```

Run tests:
```bash
pytest tests/
```

### Documentation

- Update README.md for user-facing changes
- Update API documentation for code changes
- Add inline comments for complex logic
- Update CHANGELOG.md

### Project Structure

```
src/
├── ocr/
│   ├── __init__.py
│   └── processor.py
├── embeddings/
│   ├── __init__.py
│   └── manager.py
└── rag/
    ├── __init__.py
    └── pipeline.py

tests/
├── test_ocr.py
├── test_embeddings.py
└── test_pipeline.py
```

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
feat(ocr): add batch processing support

Implement process_directory() method to handle multiple PDFs.
Includes progress tracking and error handling.

Closes #42

---

fix(embeddings): resolve memory leak in batch processing

Fixed issue where embeddings were not properly cleaned up
after processing large batches.

Fixes #55

---

docs(readme): update installation instructions

Added Windows-specific setup steps for Tesseract OCR.
```

## Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added/updated and passing
- [ ] No merge conflicts
- [ ] CHANGELOG.md updated

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested your changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] No merge conflicts

## Related Issues
Closes #(issue number)
```

### Review Process

1. At least one maintainer reviews the PR
2. All CI checks must pass
3. All review comments addressed
4. PR approved and merged by maintainer

### After Merge

- Your PR will be included in the next release
- You'll be added to contributors list
- Thank you for your contribution! 🎉

## Questions?

Feel free to reach out:

- **GitHub Discussions**: For general questions
- **GitHub Issues**: For bug reports and feature requests
- **Email**: maintainer@example.com

---

**Thank you for contributing to RAG System! 🚀**
