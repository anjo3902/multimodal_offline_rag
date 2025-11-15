# Model Management Guide

## Why Models Are NOT in Git Repository

**IMPORTANT**: LLM models are **NOT** and **SHOULD NOT** be stored in this Git repository.

### Reasons:

1. **Size**: Model files are extremely large (4.4GB+ for Mistral 7B)
2. **Git Performance**: Large binary files severely degrade Git performance
3. **Storage Costs**: GitHub has file size limits and storage quotas
4. **Best Practice**: Models are distributed through dedicated channels (Hugging Face, Ollama)
5. **Updates**: Models are frequently updated, version control creates massive repo bloat

## How to Get Models

### For Local Development (Recommended)

This project uses **Ollama** for model management:

```powershell
# 1. Install Ollama
# Download from: https://ollama.ai/download

# 2. Pull the Mistral model
ollama pull mistral

# 3. Verify installation
ollama list
```

The application will automatically use Ollama's model through the API - no manual model file management needed!

### Required Models

1. **LLM (Text Generation)**: `mistral` via Ollama
   - Size: ~4.4GB
   - Used for: Question answering, text generation

2. **Text Embeddings**: `BAAI/bge-small-en-v1.5`
   - Size: ~133MB
   - Automatically downloaded by sentence-transformers
   - Location: `~/.cache/huggingface/`

3. **Image Embeddings**: `openai/clip-vit-base-patch32`
   - Size: ~600MB
   - Automatically downloaded by transformers
   - Location: `~/.cache/huggingface/`

4. **Audio Transcription**: `Systran/faster-whisper-small`
   - Size: ~466MB
   - Automatically downloaded by faster-whisper
   - Location: `~/.cache/huggingface/`

## For Production/Alternative Deployment

### Option 1: Ollama (Recommended)
Already configured in the codebase. Just install Ollama and pull models.

### Option 2: Git LFS (Large File Storage)
If you absolutely need to version control model files:

```powershell
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.bin"
git lfs track "*.pth"
git lfs track "*.gguf"
git lfs track "*.safetensors"

# Commit .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking for models"
```

**Warning**: Git LFS has bandwidth limits on GitHub. Consider other options.

### Option 3: External Model Hosting
- **Hugging Face Hub**: Host models on HF and download at runtime
- **Cloud Storage**: AWS S3, Azure Blob, Google Cloud Storage
- **Self-hosted**: Set up your own model registry

## Configuration

Model paths are configured in `.env`:

```env
# LLM Model (Ollama model name)
LLAMA_MODEL_PATH=mistral

# Embedding device
EMBED_DEVICE=cuda  # or 'cpu' if no GPU

# Whisper model size
WHISPER_MODEL=small  # or 'tiny', 'base', 'medium', 'large'
```

## Checking Model Storage

All models are downloaded to:
- **Windows**: `C:\Users\<username>\.cache\huggingface\`
- **Linux/Mac**: `~/.cache/huggingface/`

Check your disk space:
```powershell
# Windows
Get-ChildItem ~\.cache\huggingface -Recurse | Measure-Object -Property Length -Sum

# Linux/Mac
du -sh ~/.cache/huggingface/
```

## Model Updates

To update models:
```powershell
# Update Ollama model
ollama pull mistral

# Clear Hugging Face cache (will re-download)
Remove-Item ~\.cache\huggingface\hub -Recurse -Force
```

## Troubleshooting

### "Model not found" error
```powershell
# Ensure Ollama is running
ollama list

# If model missing
ollama pull mistral
```

### Out of disk space
Models require:
- Mistral 7B: ~4.4GB
- BGE embeddings: ~133MB
- CLIP: ~600MB
- Whisper: ~466MB
- **Total**: ~5.6GB minimum

### Slow first run
First run downloads all models automatically. This is normal and only happens once.

## What's Tracked in Git

✅ **Tracked** (Source code):
- `backend/models/__init__.py`
- `backend/models/generator.py`
- `backend/models/llama_query.py`

❌ **NOT Tracked** (Model weights):
- `*.bin` (PyTorch binary)
- `*.pth` (PyTorch weights)
- `*.gguf` (GGML format)
- `*.safetensors` (Safe tensors)
- `*.onnx` (ONNX format)

These extensions are in `.gitignore` to prevent accidental commits.

## Summary

✅ **Models are managed through Ollama and Hugging Face**
✅ **No model files in Git repository**
✅ **Automatic downloading on first run**
✅ **Simple setup: Just install Ollama and run `ollama pull mistral`**

For more setup details, see [SETUP.md](SETUP.md)
