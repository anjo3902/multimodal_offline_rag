# 🚀 Setup Guide - Multimodal RAG System

Complete setup instructions for getting the Multimodal RAG system running on your machine.

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS 11+
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **Internet**: Required for initial model downloads

### Recommended (for GPU acceleration)
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CUDA**: 11.8 or 12.x
- **cuDNN**: Compatible with your CUDA version

---

## 🔧 Installation Steps

### Step 1: Clone Repository

```bash
git clone https://github.com/anjo3902/multimodal_offline_rag.git
cd multimodal_offline_rag
```

### Step 2: Set Up Python Environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r config/requirements.txt
```

**Note**: This will download ~2-3GB of packages. Be patient!

### Step 4: Install Ollama

**Windows/Mac:**
1. Download installer from https://ollama.ai
2. Run the installer
3. Verify installation: `ollama --version`

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 5: Download LLM Model

```bash
# Download Mistral 7B (recommended, 4.4GB)
ollama pull mistral

# OR download Phi-3 (smaller, 2.3GB)
ollama pull phi3

# Verify model is installed
ollama list
```

### Step 6: Configure Environment

1. **Copy example config:**
```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

2. **Edit `.env` file** with your settings:
```env
LLAMA_MODEL_PATH=mistral          # or phi3
EMBED_DEVICE=cuda                 # or cpu if no GPU
WHISPER_MODEL=small               # tiny/base/small/medium/large
```

### Step 7: Create Data Directories

```bash
# Windows
mkdir data\chroma_db data\uploads data\logs

# Linux/Mac
mkdir -p data/{chroma_db,uploads,logs}
```

### Step 8: GPU Setup (Optional but Recommended)

**Windows with NVIDIA GPU:**
1. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Download [cuDNN](https://developer.nvidia.com/cudnn)
3. Extract cuDNN to `C:\Program Files\NVIDIA\CUDNN\`
4. Update `.env`:
```env
CUDNN_PATH=C:\Program Files\NVIDIA\CUDNN\v9.5\bin\12.6
```

**Linux with NVIDIA GPU:**
```bash
# Install CUDA
sudo apt install nvidia-cuda-toolkit

# Verify
nvcc --version

# Update .env
export CUDNN_PATH=/usr/local/cuda/lib64
```

---

## 🎯 Starting the Server

### Quick Start
```bash
python run_server.py
```

### Expected Output
```
============================================================
Starting Multimodal RAG Server on http://127.0.0.1:8000
============================================================

Server is running. Press CTRL+C to stop.

Open your browser to: http://127.0.0.1:8000

Fixed issues:
  - FFmpeg added to PATH for audio processing
  - Tesseract OCR errors handled gracefully
  - Images will be indexed without OCR text if Tesseract not installed
============================================================

[OK] Added cuDNN to PATH: ...
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Access the Application
Open your browser to: **http://127.0.0.1:8000**

---

## ✅ Verification Tests

### Test 1: Check Server Status
```bash
curl http://127.0.0.1:8000/stats/
```

**Expected Response:**
```json
{
  "total_chunks": 0,
  "unique_files": 0,
  "file_types": {}
}
```

### Test 2: Upload a Test File
1. Open browser to http://127.0.0.1:8000
2. Click "Choose Files"
3. Upload a text file or PDF
4. Verify success message

### Test 3: Try a Query
1. Go to "Text Query" tab
2. Type: "What is in the uploaded document?"
3. Press Enter
4. Verify answer appears with citations [1][2][3]

---

## 🐛 Troubleshooting

### Issue: "Port 8000 already in use"
```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# Linux/Mac
lsof -i :8000
kill -9 <PID>
```

### Issue: "Module not found"
```bash
# Reinstall dependencies
pip install -r config/requirements.txt --force-reinstall
```

### Issue: "Ollama model not found"
```bash
# List installed models
ollama list

# Pull the model again
ollama pull mistral
```

### Issue: "CUDA out of memory"
1. Close other GPU applications
2. Reduce model size: Use `phi3` instead of `mistral`
3. Or run on CPU:
```env
EMBED_DEVICE=cpu
```

### Issue: "Empty transcription from audio"
- Ensure audio file is not corrupted
- Try different audio format (MP3/WAV)
- Check microphone permissions in browser settings

### Issue: "No OCR text from images"
- Install Tesseract: https://github.com/tesseract-ocr/tesseract
- Or use EasyOCR (already in requirements)
- System falls back gracefully if OCR unavailable

---

## 🔄 Updating

To pull latest changes from GitHub:

```bash
# Save your .env file first!
git pull origin main

# Reinstall dependencies if needed
pip install -r config/requirements.txt --upgrade

# Restart server
python run_server.py
```

---

## 📚 Next Steps

1. **Upload Documents**: Try PDFs, Word docs, images
2. **Voice Recording**: Test audio queries with microphone
3. **Check Advanced Features**: 
   - Citation transparency
   - Confidence scoring
   - Source metadata

For detailed API documentation, see [README.md](README.md)

---

## 💬 Getting Help

- **Issues**: https://github.com/anjo3902/multimodal_offline_rag/issues
- **Discussions**: https://github.com/anjo3902/multimodal_offline_rag/discussions

---

**Happy querying! 🎉**
