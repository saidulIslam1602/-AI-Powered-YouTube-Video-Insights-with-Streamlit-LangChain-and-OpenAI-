# Contributing to YouTube Video Insights

Thank you for your interest in contributing to YouTube Video Insights! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- OpenAI API key
- Basic knowledge of Python, Streamlit, and LangChain

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/youtube-video-insights.git
   cd youtube-video-insights
   ```

2. **Set Up Environment**
   ```bash
   # Quick setup using Make
   make setup
   
   # Or manual setup
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,docs]"
   pre-commit install
   ```

3. **Configure Environment**
   ```bash
   cp env.example .env
   # Edit .env with your OpenAI API key
   ```

4. **Verify Setup**
   ```bash
   make test
   make run
   ```

## Development Workflow

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical production fixes

### Making Changes

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   make check  # Runs linting and tests
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create a Pull Request on GitHub
   ```

## Coding Standards

### Python Style Guide
- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use descriptive variable and function names

### Code Formatting
We use automated tools for consistent formatting:

```bash
# Format code
make format

# Check formatting
make lint
```

### Import Organization
```python
# Standard library imports
import os
import sys
from typing import List, Dict

# Third-party imports
import streamlit as st
from langchain import OpenAI

# Local imports
from src.config.settings import settings
from src.utils.logger import logger
```

### Documentation Standards
- Use Google-style docstrings
- Document all public functions and classes
- Include type hints
- Provide usage examples for complex functions

```python
def process_video(url: str, use_cache: bool = True) -> ProcessedVideo:
    """Process a YouTube video and extract insights.
    
    Args:
        url: YouTube video URL to process
        use_cache: Whether to use cached results if available
        
    Returns:
        ProcessedVideo object containing transcript and vector store
        
    Raises:
        InvalidVideoURLError: If the URL is not a valid YouTube URL
        TranscriptError: If transcript cannot be fetched
        
    Example:
        >>> processor = VideoProcessor()
        >>> video = processor.process_video("https://youtube.com/watch?v=...")
    """
```

## Testing Guidelines

### Test Structure
- Unit tests: Test individual functions/methods
- Integration tests: Test component interactions
- End-to-end tests: Test complete workflows

### Writing Tests
```python
# Test file naming: test_<module_name>.py
# Test class naming: Test<ClassName>
# Test method naming: test_<method_name>_<scenario>

class TestVideoProcessor:
    def test_extract_video_id_valid_url(self):
        """Test video ID extraction from valid URL."""
        processor = VideoProcessor()
        video_id = processor.extract_video_id("https://youtube.com/watch?v=123")
        assert video_id == "123"
```

### Running Tests
```bash
# All tests
make test

# Specific test file
pytest tests/test_video_processor.py

# Specific test method
pytest tests/test_video_processor.py::TestVideoProcessor::test_extract_video_id_valid_url

# With coverage
make test

# Fast tests (no coverage)
make test-fast
```

## Commit Message Guidelines

We follow the Conventional Commits specification:

### Format
```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

### Examples
```bash
feat(video-processor): add caching for processed videos
fix(ui): resolve responsive design issues on mobile
docs(readme): update installation instructions
test(query-engine): add tests for language detection
```

## Architecture Guidelines

### Project Structure
```
src/
├── config/           # Configuration management
├── models/           # Core business logic
├── ui/              # User interface components
├── utils/           # Utilities and helpers
└── __init__.py
```

### Adding New Features

1. **Model Classes** (`src/models/`)
   - Core business logic
   - Data processing
   - External API interactions

2. **UI Components** (`src/ui/`)
   - Streamlit interface components
   - User interaction handling

3. **Utilities** (`src/utils/`)
   - Helper functions
   - Common utilities
   - Logging and error handling

4. **Configuration** (`src/config/`)
   - Settings management
   - Environment variables

### Error Handling
- Use custom exceptions from `src/utils/exceptions.py`
- Provide meaningful error messages
- Log errors appropriately
- Handle edge cases gracefully

```python
try:
    result = process_video(url)
except InvalidVideoURLError as e:
    logger.error(f"Invalid URL: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise ProcessingError(f"Failed to process video: {e}")
```

## Documentation

### Code Documentation
- Use docstrings for all public functions and classes
- Include type hints
- Provide examples for complex functions

### User Documentation
- Update README.md for user-facing changes
- Add examples for new features
- Update configuration documentation

### API Documentation
- Document all public APIs
- Include parameter descriptions
- Provide usage examples

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - Python version
   - Operating system
   - Dependency versions

2. **Steps to Reproduce**
   - Minimal reproducible example
   - Expected vs actual behavior
   - Error messages and stack traces

3. **Additional Context**
   - Screenshots (if applicable)
   - Log files
   - Configuration details

## Feature Requests

When requesting features:

1. **Describe the Problem**
   - What problem does this solve?
   - Who would benefit?

2. **Propose a Solution**
   - Detailed description
   - Alternative solutions considered
   - Implementation approach

3. **Additional Context**
   - Use cases
   - Examples
   - Related issues

## Security

### Reporting Security Issues
- **DO NOT** open public issues for security vulnerabilities
- Email security@yourproject.com
- Provide detailed information
- Allow time for investigation and fix

### Security Best Practices
- Never commit API keys or secrets
- Use environment variables for configuration
- Validate all user inputs
- Follow principle of least privilege

## Pull Request Process

### Before Submitting
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] No merge conflicts

### PR Requirements
- [ ] Descriptive title and description
- [ ] Link to related issues
- [ ] Screenshots (if UI changes)
- [ ] Test coverage maintained
- [ ] Documentation updated

### Review Process
1. Automated checks must pass
2. Code review by maintainers
3. Manual testing (if needed)
4. Approval and merge

## 🎉 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## ❓ Questions

- 📖 [Documentation](https://yourusername.github.io/youtube-video-insights)
- 💬 [Discussions](https://github.com/yourusername/youtube-video-insights/discussions)
- 🐛 [Issues](https://github.com/yourusername/youtube-video-insights/issues)

Thank you for contributing to YouTube Video Insights! 🎥✨ 