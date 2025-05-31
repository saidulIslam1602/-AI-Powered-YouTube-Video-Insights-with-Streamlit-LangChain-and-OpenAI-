# 🎥 YouTube Video Insights

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/framework-Streamlit-red.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/AI-OpenAI%20GPT--4-green.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, AI-powered web application that extracts and analyzes insights from YouTube video transcripts using advanced language models and vector search technology.

## 🚀 Features

### Core Functionality
- **🎯 Advanced Video Analysis**: Extract and analyze insights from YouTube video transcripts
- **🌐 Multi-language Support**: Automatic language detection for both videos and queries
- **🔍 Semantic Search**: FAISS-powered vector search for relevant content discovery
- **💬 Natural Language Queries**: Ask questions in natural language and get contextual answers
- **📊 Confidence Scoring**: AI-generated confidence scores for response quality assessment

### Technical Features
- **⚡ Performance Optimization**: Intelligent caching system for faster processing
- **🛡️ Error Handling**: Comprehensive error handling with custom exceptions
- **📝 Logging**: Professional logging system with multiple levels and file output
- **🔧 Configuration Management**: Centralized configuration using Pydantic settings
- **🧪 Testing**: Comprehensive test suite with pytest and coverage reporting
- **📚 Documentation**: Detailed API documentation and user guides

### User Experience
- **🎨 Modern UI**: Beautiful, responsive Streamlit interface with custom styling
- **💡 Smart Suggestions**: AI-generated question suggestions based on video content
- **📈 Analytics**: Processing time tracking and performance metrics
- **🔄 Cache Management**: Built-in cache controls for optimal performance
- **📱 Mobile Friendly**: Responsive design that works on all devices

## 🏗️ Architecture

```
src/
├── config/           # Configuration management
├── models/           # Core business logic
│   ├── video_processor.py  # Video processing and transcript handling
│   └── query_engine.py     # Query processing and response generation
├── ui/              # User interface components
├── utils/           # Utilities and helpers
└── __init__.py      # Package initialization
```

## 📋 Requirements

- **Python**: 3.8 or higher
- **OpenAI API Key**: Required for GPT-4 access
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 1GB free space for dependencies and cache

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/youtube-video-insights.git
cd youtube-video-insights
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Or install with development dependencies
pip install -e ".[dev,docs]"
```

### 4. Environment Configuration
```bash
# Copy example environment file
cp env.example .env

# Edit .env file with your OpenAI API key
# OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Verify Installation
```bash
# Run tests to verify everything is working
pytest

# Run the application
streamlit run app.py
```

## 🚀 Quick Start

### Basic Usage

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Process a Video**:
   - Enter a YouTube URL (e.g., `https://www.youtube.com/watch?v=VIDEO_ID`)
   - Click "🔄 Process Video"
   - Wait for processing to complete

4. **Ask Questions**:
   - Use suggested questions or type your own
   - Get AI-powered answers with confidence scores
   - View source chunks and processing details

### Example Queries

```
"What are the main topics discussed in this video?"
"Can you summarize the key points made by the speaker?"
"What examples are given to support the main argument?"
"Are there any specific statistics or data mentioned?"
"What conclusions does the presenter reach?"
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 access | - | ✅ |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4-turbo` | ❌ |
| `MAX_TOKENS` | Maximum tokens for responses | `2000` | ❌ |
| `TEMPERATURE` | Model temperature (0-1) | `0.1` | ❌ |
| `CHUNK_SIZE` | Text chunk size for processing | `1000` | ❌ |
| `CHUNK_OVERLAP` | Overlap between chunks | `100` | ❌ |
| `SIMILARITY_K` | Number of similar chunks to retrieve | `4` | ❌ |
| `MAX_VIDEO_LENGTH_MINUTES` | Maximum video length (minutes) | `180` | ❌ |
| `CACHE_ENABLED` | Enable caching | `true` | ❌ |
| `CACHE_TTL_SECONDS` | Cache time-to-live | `3600` | ❌ |
| `LOG_LEVEL` | Logging level | `INFO` | ❌ |

### Advanced Configuration

For advanced users, you can modify settings in `src/config/settings.py`:

```python
# Example: Custom model configuration
OPENAI_MODEL = "gpt-4-turbo"
TEMPERATURE = 0.2
MAX_TOKENS = 3000

# Example: Processing optimization
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
SIMILARITY_K = 6
```

## 🧪 Testing

### Run All Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Test Structure
```
tests/
├── test_video_processor.py    # Video processing tests
├── test_query_engine.py       # Query engine tests
├── test_config.py             # Configuration tests
└── integration/               # Integration tests
```

## 🔍 Code Quality

### Linting and Formatting
```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## 📊 Performance

### Caching Strategy
- **Video Processing**: Cached by video ID and processing parameters
- **Embeddings**: Cached to avoid recomputation
- **Cache TTL**: Configurable (default: 1 hour)

### Performance Metrics
- **Processing Time**: Tracked for each operation
- **Memory Usage**: Optimized vector storage
- **Response Time**: Sub-second query responses (cached videos)

### Optimization Tips
1. **Enable Caching**: Reduces processing time by 90%+
2. **Adjust Chunk Size**: Larger chunks = fewer API calls
3. **Limit Video Length**: Shorter videos process faster
4. **Use Appropriate Model**: Balance quality vs. speed

## 🚢 Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t youtube-insights .

# Run container
docker run -p 8501:8501 --env-file .env youtube-insights
```

### Cloud Deployment

#### Streamlit Cloud
1. Fork this repository
2. Connect to Streamlit Cloud
3. Add secrets in dashboard
4. Deploy with one click

#### Heroku
```bash
# Install Heroku CLI and login
heroku login

# Create Heroku app
heroku create your-app-name

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key_here

# Deploy
git push heroku main
```

#### AWS/GCP/Azure
See `deployment/` directory for cloud-specific configurations.

## 🛠️ Development

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run in development mode
streamlit run app.py --server.runOnSave true
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Adding New Features
1. **Add Model Classes**: In `src/models/`
2. **Add Tests**: In `tests/`
3. **Update Configuration**: In `src/config/`
4. **Update Documentation**: In `README.md` and docstrings

## 📄 API Reference

### VideoProcessor Class
```python
from src.models.video_processor import VideoProcessor

processor = VideoProcessor()
processed_video = processor.process_video(url="https://youtube.com/...")
```

### QueryEngine Class
```python
from src.models.query_engine import QueryEngine

engine = QueryEngine()
result = engine.generate_response(processed_video, query="What is this about?")
```

## 🐛 Troubleshooting

### Common Issues

#### OpenAI API Errors
```bash
Error: OpenAI API key is not set
Solution: Ensure OPENAI_API_KEY is set in .env file
```

#### YouTube Transcript Errors
```bash
Error: No transcript available
Solution: Try a different video with available transcripts
```

#### Memory Issues
```bash
Error: Out of memory during processing
Solution: Reduce chunk size or video length limit
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
streamlit run app.py
```

### Getting Help
- 📖 [Documentation](https://yourusername.github.io/youtube-video-insights)
- 🐛 [Issue Tracker](https://github.com/yourusername/youtube-video-insights/issues)
- 💬 [Discussions](https://github.com/yourusername/youtube-video-insights/discussions)

## 📈 Roadmap

### Upcoming Features
- [ ] **Batch Processing**: Multiple videos at once
- [ ] **Export Options**: PDF/Word report generation
- [ ] **Audio Processing**: Direct audio file support
- [ ] **Custom Models**: Support for other LLM providers
- [ ] **API Endpoints**: REST API for programmatic access
- [ ] **Database Integration**: Persistent storage options
- [ ] **User Authentication**: Multi-user support
- [ ] **Analytics Dashboard**: Usage statistics and insights

### Version History
- **v1.0.0** (Current): Production-ready release with full features
- **v0.3.0**: Added caching and performance optimizations
- **v0.2.0**: Enhanced UI and error handling
- **v0.1.0**: Initial basic functionality

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT-4 language model
- **Streamlit** for the web framework
- **LangChain** for LLM orchestration
- **FAISS** for vector similarity search
- **YouTube Transcript API** for transcript access

## 📞 Support

If you encounter any issues or have questions:

1. Check the [FAQ](docs/faq.md)
2. Search existing [issues](https://github.com/yourusername/youtube-video-insights/issues)
3. Create a new issue with detailed information
4. Join our [community discussions](https://github.com/yourusername/youtube-video-insights/discussions)

---

**Built with ❤️ by [Your Name](https://github.com/yourusername)**

*Transform any YouTube video into actionable insights with the power of AI!*

