# 🔍 Multimodal RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)

A production-ready Multimodal Retrieval-Augmented Generation (RAG) system that can ingest, index, and query diverse data formats (Documents, Images, Audio) using offline Large Language Models.

## ✨ Features

### Core Capabilities
- **Multimodal Ingestion**: PDF, DOCX, TXT, Images (PNG/JPG), Audio (MP3/WAV/M4A/FLAC)
- **OCR Integration**: EasyOCR with GPU acceleration for text extraction from images
- **Speech-to-Text**: Faster-Whisper for high-speed audio transcription
- **Vector Database**: ChromaDB for semantic search across all modalities
- **Offline LLM**: Mistral 7B via Ollama (no internet required)
- **GPU Optimized**: Full CUDA acceleration for faster responses
- **Modern UI**: Clean, responsive web interface with voice recording

### Advanced Features
- **Citation Transparency**: Numbered inline citations [1][2][3] for every factual claim
- **Confidence Scoring**: Answer reliability metrics (0-100%) with quality indicators
- **Quality Scoring**: Relevance percentages for each source
- **Voice Recording**: Direct browser recording with real-time transcription
- **Cross-Format Search**: Text embeddings (BGE-small-en) + CLIP visual embeddings
- **Source Metadata**: File size, creation date, OCR confidence, audio duration
- **Statistics Dashboard**: Real-time metrics on indexed content

### Search Modes
1. **Text Query** - Natural language questions with semantic search
2. **Image Query** - Upload images/screenshots, extract text via OCR, find related content
3. **Audio Query** - Upload voice recordings or record directly in browser

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended for speed)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/anjo3902/multimodal_offline_rag.git
cd multimodal_rag_free
```

2. **Create virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r config/requirements.txt
```

4. **Install Ollama and download Mistral 7B:**
```bash
# Download Ollama from https://ollama.ai
ollama pull mistral
```

5. **Set up environment variables:**
Create `.env` file in root directory:
```env
LLAMA_MODEL_PATH=mistral
CUDNN_PATH=C:\Program Files\NVIDIA\CUDNN\v9.5\bin\12.6
```

6. **Start the server:**
```bash
python run_server.py
```

7. **Open browser:**
Navigate to `http://127.0.0.1:8000`

## 📖 Usage Guide

### 1️⃣ Upload Documents
- Click "Choose Files" and select your documents, images, or audio files
- Multiple file types supported in one upload
- System automatically processes and indexes content
- Supported formats:
  - **Documents**: PDF, DOCX, TXT
  - **Images**: PNG, JPG, JPEG (OCR enabled)
  - **Audio**: MP3, WAV, M4A, FLAC (transcribed automatically)

### 2️⃣ Query Your Data

**Text Mode:**
```
Example: "What are the impacts on job seekers?"
```
Type your question and press Enter or click "Search"

**Image Mode:**
1. Upload an image or screenshot
2. System extracts text via OCR
3. Finds semantically related content across all indexed files

**Audio Mode:**

*Option A: Record Voice (Fastest - 2-5 seconds)*
1. Click "🎤 Start Voice Recording"
2. Allow microphone permission
3. Speak your question
4. Click "⏹️ Stop Recording"
5. Click "🔍 Search by Audio"

*Option B: Upload Audio File (10-20 seconds)*
1. Select MP3/WAV/M4A/FLAC file
2. Click "Search by Audio"
3. System transcribes and searches

### 3️⃣ Interpret Results

**Answer Format:**
```
## Summary
Job seekers face increasing challenges due to AI adoption [1]. 
However, new opportunities emerge in tech fields [2].

## Key Points
- **AI tools** automate routine tasks [1]
- **Upskilling** becomes essential [2]
- **Remote work** expands opportunities [3]

## Citations
[1] Source document name
[2] Another source
[3] Third source
```

**Confidence Levels:**
- ✅ **High (≥80%)**: Green banner - Highly reliable answer
- ⚠️ **Medium (60-79%)**: Yellow banner - Moderately reliable
- ❌ **Low (<60%)**: Red banner - Limited confidence

**Quality Indicators:**
- 🟢 **Excellent (≥80%)**: Highly relevant source
- 🟡 **Good (60-79%)**: Relevant source
- 🟠 **Fair (40-59%)**: Somewhat relevant
- 🔴 **Low (<40%)**: Marginally relevant

## 🏗️ Architecture

### Directory Structure
```
multimodal_rag/
├── backend/
│   ├── api/
│   │   └── app.py              # FastAPI server & routes
│   ├── core/
│   │   ├── embeddings.py       # BGE text embeddings
│   │   ├── clip_embeddings.py  # CLIP image embeddings
│   │   ├── indexer.py          # ChromaDB indexing
│   │   ├── ingestion.py        # File processing
│   │   └── ocr_engine.py       # EasyOCR integration
│   ├── models/
│   │   ├── llama_query.py      # LLM wrapper
│   │   └── generator.py        # Mistral generation
│   └── utils/
│       ├── utils.py            # Prompt building
│       └── format_answer.py    # Citation formatting
├── frontend/
│   └── index.html              # Web UI
├── config/
│   ├── requirements.txt        # Python dependencies
│   └── .env                    # Environment variables
├── scripts/
│   ├── run_all.ps1            # Full system startup
│   ├── run_tests.ps1          # Testing automation
│   └── stop_server.ps1        # Shutdown script
├── data/                      # Runtime (gitignored)
│   ├── chroma_db/            # Vector database
│   ├── uploads/              # Uploaded files
│   └── logs/                 # Server logs
├── run_server.py             # Entry point
└── README.md                 # This file
```

### Technology Stack

**Backend:**
- **FastAPI**: High-performance API server
- **ChromaDB**: Vector database for semantic search
- **BGE-small-en**: Text embeddings (133MB)
- **CLIP ViT-B/32**: Image embeddings
- **Faster-Whisper**: Audio transcription (GPU accelerated)
- **Mistral 7B**: LLM via Ollama (4.4GB)
- **EasyOCR**: Optical Character Recognition

**Frontend:**
- Vanilla JavaScript (no framework dependencies)
- MediaRecorder API for voice recording
- Web Speech Recognition API for real-time transcription
- Responsive CSS with modern design

## ⚙️ Configuration

### Model Settings

**Text Embeddings** (`backend/core/embeddings.py`):
```python
TEXT_MODEL = "BAAI/bge-small-en"  # 133MB, high quality
```

**Image Embeddings** (`backend/core/clip_embeddings.py`):
```python
IMAGE_MODEL = "clip-ViT-B-32"  # CLIP visual encoder
```

**LLM Generation** (`backend/models/generator.py`):
```python
model = "mistral"           # Mistral 7B
temperature = 0.3           # Lower = more factual
num_predict = 800          # Max tokens (faster)
```

### Environment Variables

Create `.env` file:
```env
# LLM Model
LLAMA_MODEL_PATH=mistral

# GPU Acceleration (Windows)
CUDNN_PATH=C:\Program Files\NVIDIA\CUDNN\v9.5\bin\12.6

# Server Settings
HOST=127.0.0.1
PORT=8000
```

### GPU Optimization

For faster inference, ensure CUDA is properly configured:

**Windows:**
1. Install CUDA Toolkit 12.x from NVIDIA
2. Download cuDNN and extract to `C:\Program Files\NVIDIA\CUDNN\`
3. Set `CUDNN_PATH` in `.env`

**Linux:**
```bash
export CUDNN_PATH=/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH
```

## 🧪 Testing

### Manual Testing

1. **Start server:**
```bash
python run_server.py
```

2. **Open browser:** http://127.0.0.1:8000

3. **Test each mode:**
   - Upload test documents
   - Try text query
   - Upload test image
   - Record voice or upload audio

### Automated Testing

```bash
# Run all tests
.\scripts\run_tests.ps1

# Test specific component
python -m pytest test/
```

### Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Voice Recording | 2-5s | Real-time transcription |
| Audio File Upload | 10-20s | Includes transcription |
| Text Query | 8-15s | Mistral 7B generation |
| Image Upload | 5-10s | OCR + indexing |
| Document Upload | 3-8s | Per document |

## 🔧 Troubleshooting

### Server Won't Start

**Issue:** "Port 8000 already in use"
```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# Linux
lsof -i :8000
kill -9 <PID>
```

**Issue:** "CUDA out of memory"
- Reduce `num_predict` in `generator.py`
- Close other GPU applications
- Restart with CPU-only mode: Set `CUDA_VISIBLE_DEVICES=-1`

### Voice Recording Issues

**Issue:** Microphone permission denied
- Browser Settings → Privacy → Microphone → Allow for localhost
- Try Chrome/Edge (Firefox/Safari not fully supported)

**Issue:** Empty transcription
- Speak louder and clearer
- Check microphone is working in other apps
- Ensure WebM format supported (server logs will show format)

### Answer Quality Issues

**Issue:** Irrelevant answers
- Upload more diverse documents
- Use more specific questions
- Check confidence score (low = unreliable)

**Issue:** No citations [1][2][3]
- Check `backend/utils/utils.py` prompt template
- Verify sources were properly indexed
- Review server logs for LLM errors

### Model Issues

**Issue:** "Mistral model not found"
```bash
ollama list          # Check installed models
ollama pull mistral  # Download if missing
ollama run mistral "test"  # Verify working
```

**Issue:** Slow generation (>60 seconds)
- Verify GPU acceleration enabled
- Check CUDA/cuDNN installed
- Reduce `num_predict` in config
- Consider using smaller model

## 📊 API Reference

### Upload Files
```http
POST /upload/
Content-Type: multipart/form-data

files: [file1, file2, ...]
```

**Response:**
```json
{
  "message": "Uploaded 3 files",
  "files": ["doc1.pdf", "image.png", "audio.mp3"]
}
```

### Text Query
```http
POST /query/
Content-Type: application/json

{
  "query": "What are the key points?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "## Summary\n...",
  "sources": [
    {
      "file_name": "document.pdf",
      "chunk": "relevant text...",
      "quality_score": 92.5
    }
  ],
  "confidence_score": 87.3
}
```

### Image Query
```http
POST /query/
Content-Type: multipart/form-data

query_image: <file>
```

### Audio Query
```http
POST /query/
Content-Type: multipart/form-data

query_audio: <file>
```

### Statistics
```http
GET /stats/
```

**Response:**
```json
{
  "total_chunks": 156,
  "unique_files": 12,
  "file_types": {
    "pdf": 5,
    "png": 4,
    "mp3": 3
  }
}
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for local LLM inference
- **ChromaDB** for vector database
- **Mistral AI** for the Mistral 7B model
- **OpenAI CLIP** for multimodal embeddings
- **Faster-Whisper** for audio transcription
- **EasyOCR** for optical character recognition

## 📞 Support

For issues and questions:
- GitHub Issues: [multimodal_offline_rag/issues](https://github.com/anjo3902/multimodal_offline_rag/issues)

## 🎯 Roadmap

- [ ] Add support for video files
- [ ] Implement real-time streaming responses
- [ ] Add multi-language support
- [ ] Create Docker deployment
- [ ] Add authentication system
- [ ] Implement conversation history
- [ ] Add export to PDF/Word features

---

**Built with ❤️ for offline multimodal intelligence**
