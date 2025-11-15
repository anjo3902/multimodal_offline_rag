"""Run the server - keeps it alive"""
import uvicorn
import os

# Add FFmpeg to PATH
if os.path.exists("C:\\ffmpeg\\bin"):
    os.environ["PATH"] = "C:\\ffmpeg\\bin;" + os.environ.get("PATH", "")

print("="*60)
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "127.0.0.1")
print(f"Starting Multimodal RAG Server on http://{HOST}:{PORT}")
print("="*60)
print("\nServer is running. Press CTRL+C to stop.")
print("\nOpen your browser to: http://127.0.0.1:8000")
print("\nFixed issues:")
print("  - FFmpeg added to PATH for audio processing")
print("  - Tesseract OCR errors handled gracefully")
print("  - Images will be indexed without OCR text if Tesseract not installed")
print("="*60)
print()

if __name__ == "__main__":
    try:
        uvicorn.run("backend.api.app:app", host=HOST, port=PORT, log_level="info")
    except OSError as e:
        # Common issue on Windows: port already in use
        print(f"\n[ERROR] Could not start server: {e}")
        if '10048' in str(e) or 'Address already in use' in str(e) or 'Only one usage of each socket address' in str(e):
            print("It looks like the port is already in use. Try one of the following:")
            print("  - Stop the process using the port (see instructions below)")
            print("  - Run with a different port: set PORT env var, e.g. `set PORT=7860` then run again")
            print("Windows command to find PID listening on port 8000:")
            print("  netstat -ano | findstr :8000")
            print("Then kill it: taskkill /F /PID <pid>")
        raise
