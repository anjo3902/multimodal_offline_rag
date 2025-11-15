# app.py
import os
import re
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
from backend.core.embeddings import Embedder
from backend.core.indexer import ChromaIndexer
from backend.models.llama_query import generate_answer
from backend.utils.utils import build_prompt
from backend.utils.format_answer import format_with_sources
from dotenv import load_dotenv
import numpy as np
import shutil
from pathlib import Path
from backend.core.ingestion import load_image, transcribe_audio_whisper, extract_text_from_pdf, extract_text_from_docx, chunk_text, chunk_audio_by_seconds
import uuid
from backend.core.clip_embeddings import get_clip_embedder
from backend.core.ocr_engine import get_ocr_engine

load_dotenv()

# Add FFmpeg to PATH if it exists
if os.path.exists("C:\\ffmpeg\\bin"):
    os.environ["PATH"] = "C:\\ffmpeg\\bin;" + os.environ.get("PATH", "")

# Add cuDNN to PATH for Faster-Whisper GPU acceleration
try:
    import nvidia.cudnn
    from pathlib import Path
    cudnn_path = Path(nvidia.cudnn.__path__[0])
    cudnn_bin = str(cudnn_path / "bin")
    if os.path.exists(cudnn_bin):
        os.environ["PATH"] = cudnn_bin + ";" + os.environ.get("PATH", "")
        print(f"[OK] Added cuDNN to PATH: {cudnn_bin}")
except Exception as e:
    print(f"[WARNING] Could not add cuDNN to PATH: {e}")

EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "/path/to/llama.bin")
TOP_K_DEFAULT = 5

app = FastAPI(title="Multimodal RAG (Offline)")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Initialize embedders and OCR
embedder = Embedder(device=EMBED_DEVICE)  # For text/audio (sentence-transformers)
clip_embedder = None  # Lazy load CLIP for images (GPU intensive)
ocr_engine = None  # Lazy load OCR engine
indexer = ChromaIndexer()

@app.get("/", response_class=HTMLResponse)
def home():
    try:
        # Navigate from backend/api/ to frontend/
        html_path = Path(__file__).parent.parent.parent / "frontend" / "index.html"
        return html_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error loading index.html: {e}")
        return HTMLResponse(content=f"<h1>Error loading page: {e}</h1>", status_code=500)

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and index files (PDF, DOCX, images, audio)"""
    saved = []
    file_details = []  # Store details about each file including OCR text
    os.makedirs("data/uploads", exist_ok=True)
    
    for f in files:
        try:
            print(f"\n=== Uploading file: {f.filename} ===")
            dest = os.path.join("data/uploads", f"{uuid.uuid4().hex}_{f.filename}")
            
            # Save file first
            with open(dest, "wb") as out:
                content = await f.read()
                out.write(content)
            print(f"File saved to: {dest}")
            
            # Get extension
            ext = Path(dest).suffix.lower()
            print(f"File extension: {ext}")
            
            # Process based on file type
            if ext in [".pdf", ".docx", ".txt"]:
                print("Processing document...")
                # reuse ingestion / indexing flow: read text, chunk, embed, add
                if ext == ".pdf":
                    text = extract_text_from_pdf(dest)
                    doc_type = "PDF"
                elif ext == ".docx":
                    text = extract_text_from_docx(dest)
                    doc_type = "DOCX"
                else:
                    text = Path(dest).read_text(encoding="utf-8")
                    doc_type = "TXT"
                
                print(f"  [OK] Extracted {len(text)} characters from {doc_type}")
                
                chunks = chunk_text(text, chunk_size=800, overlap=100)
                ids, embeddings, metadatas, documents = [], [], [], []
                for i, (chunk,s,e) in enumerate(chunks):
                    doc_id = f"{Path(dest).name}::chunk::{i}::{uuid.uuid4().hex[:8]}"
                    emb = embedder.embed_text(chunk)[0]
                    ids.append(doc_id)
                    embeddings.append(emb)
                    metadatas.append({"source": Path(dest).name, "path": dest, "chunk": i, "start": s, "end": e, "type":"document"})
                    documents.append(chunk)
                indexer.add_items(ids, np.vstack(embeddings), metadatas, documents)
                print(f"  [OK] Document indexed: {len(ids)} chunks")
                
                # Store document info for response
                file_info = {
                    "filename": f.filename,
                    "type": "document",
                    "doc_type": doc_type,
                    "path": dest,
                    "extracted_text": text,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "chunks": len(chunks)
                }
                file_details.append(file_info)
                
            elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                print("Processing image with CLIP + OCR...")
                
                # Lazy load CLIP embedder
                global clip_embedder, ocr_engine
                if clip_embedder is None:
                    print("  [CLIP] Loading CLIP model for visual embeddings...")
                    clip_embedder = get_clip_embedder(device=EMBED_DEVICE)
                
                # Lazy load OCR engine
                if ocr_engine is None:
                    print("  [NOTE] Initializing OCR engine...")
                    ocr_engine = get_ocr_engine()
                
                # Load image
                pil = load_image(dest)
                filename_without_ext = Path(dest).stem
                import re
                cleaned_name = re.sub(r'[0-9a-f]{8,}', '', filename_without_ext)
                cleaned_name = re.sub(r'_+', ' ', cleaned_name)
                cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()

                ids, embeddings, metadatas, documents = [], [], [], []
                
                # Strategy 1: CLIP visual embedding (ALWAYS - works for all images)
                print("  [CLIP] Creating CLIP visual embedding...")
                visual_emb = clip_embedder.embed_image(pil)
                visual_doc_id = f"{Path(dest).name}::clip_visual::{uuid.uuid4().hex[:8]}"
                visual_description = f"Image: {cleaned_name}. Visual content including charts, diagrams, photos, screenshots, or graphics."
                
                ids.append(visual_doc_id)
                embeddings.append(visual_emb)
                metadatas.append({
                    "source": Path(dest).name,
                    "path": dest,
                    "file_path": os.path.abspath(dest),
                    "type": "image",
                    "embedding_type": "clip_visual",
                    "description": visual_description
                })
                documents.append(visual_description)
                print(f"  [OK] CLIP visual embedding created (512-dim)")
                
                # Strategy 2: Advanced OCR text extraction
                print("  [NOTE] Extracting text with OCR...")
                ocr_text = ""
                ocr_backend = None
                if ocr_engine.is_available():
                    ocr_text = ocr_engine.extract_text(pil)
                    ocr_backend = ocr_engine.get_backend()
                else:
                    print("  [WARNING]  No OCR backend available - install easyocr for best results")
                
                # Store OCR results for response
                file_info = {
                    "filename": f.filename,
                    "type": "image",
                    "path": dest,
                    "ocr_text": ocr_text if ocr_text else None,
                    "ocr_backend": ocr_backend,
                    "ocr_char_count": len(ocr_text) if ocr_text else 0,
                    "visual_embedding": True
                }
                
                if ocr_text and ocr_text.strip():
                    print(f"  [OK] OCR extracted {len(ocr_text)} characters using {ocr_backend}")
                    print(f"  [DOC] Extracted text preview: {ocr_text[:200]}...")
                    # Chunk OCR text and create CLIP text embeddings
                    chunks = chunk_text(ocr_text.strip(), chunk_size=800, overlap=100)
                    file_info["text_chunks"] = len(chunks)
                    for i, (chunk, s, e) in enumerate(chunks):
                        # Use CLIP to embed the text (unified space with images)
                        text_emb = clip_embedder.embed_text(chunk)
                        doc_id = f"{Path(dest).name}::clip_text::{i}::{uuid.uuid4().hex[:8]}"
                        ids.append(doc_id)
                        embeddings.append(text_emb)
                        metadatas.append({
                            "source": Path(dest).name,
                            "path": dest,
                            "file_path": os.path.abspath(dest),
                            "type": "image",
                            "embedding_type": "clip_text",
                            "ocr_text": chunk,
                            "chunk": i,
                            "char_start": s,
                            "char_end": e,
                            "description": f"OCR text from image: {cleaned_name}"
                        })
                        documents.append(chunk)
                    print(f"  [OK] Indexed {len(chunks)} OCR text chunks with CLIP embeddings")
                else:
                    print(f"  [INFO]  No text detected in image (pure visual content)")
                    file_info["text_chunks"] = 0
                
                # Add all embeddings to CLIP collection (512-dim)
                if embeddings:
                    indexer.add_items(ids, np.vstack(embeddings), metadatas, documents, use_clip=True)
                    print(f"  [OK] Image indexed with {len(ids)} embeddings (visual + text) in CLIP collection")
                
                file_details.append(file_info)                
            elif ext in [".wav", ".mp3", ".m4a", ".flac"]:
                print("Processing audio...")
                # chunk audio and transcribe
                chunks = chunk_audio_by_seconds(dest, chunk_seconds=30, overlap_seconds=5)
                print(f"Audio split into {len(chunks)} chunks")
                ids, embeddings, metadatas, documents = [], [], [], []
                
                all_transcriptions = []  # Store all transcribed text
                
                for idx, (tmp_path, start, end) in enumerate(chunks):
                    try:
                        print(f"Transcribing chunk {idx+1}/{len(chunks)} ({start:.1f}s - {end:.1f}s)...")
                    except:
                        pass
                    
                    res = transcribe_audio_whisper(tmp_path, model_name=WHISPER_MODEL, device=EMBED_DEVICE)
                    text = res.get("text","")
                    
                    if not text.strip():
                        print(f"  Chunk {idx+1}: No transcription (empty)")
                        continue
                    
                    print(f"  Chunk {idx+1}: Transcribed {len(text)} characters")
                    all_transcriptions.append(text.strip())
                    
                    emb = embedder.embed_text(text)[0]
                    doc_id = f"{Path(dest).name}::audio::{int(start)}-{int(end)}::{uuid.uuid4().hex[:8]}"
                    ids.append(doc_id)
                    embeddings.append(emb)
                    metadatas.append({"source": Path(dest).name, "path": dest, "type":"audio", "start": start, "end": end, "chunk_file": tmp_path})
                    documents.append(text)
                
                if ids:
                    print(f"Adding {len(ids)} audio chunks to index...")
                    indexer.add_items(ids, np.vstack(embeddings), metadatas, documents)
                    print(f"  [OK] Audio indexing complete!")
                    
                    # Store audio transcription info for response
                    full_transcription = " ".join(all_transcriptions)
                    file_info = {
                        "filename": f.filename,
                        "type": "audio",
                        "path": dest,
                        "transcribed_text": full_transcription,
                        "char_count": len(full_transcription),
                        "word_count": len(full_transcription.split()),
                        "chunks": len(ids),
                        "duration_seconds": round(chunks[-1][2], 1) if chunks else 0
                    }
                    file_details.append(file_info)
                    print(f"  [NOTE] Audio file_info created: {file_info['filename']}, {file_info['char_count']} chars, {file_info['duration_seconds']}s")
                else:
                    print("Warning: No audio chunks were transcribed successfully")
                    
            else:
                print(f"Unsupported file type: {ext}")
            
            saved.append(dest)
            print(f"=== File processed: {f.filename} ===\n")
            
        except Exception as e:
            error_msg = f"Error processing {f.filename}: {str(e)}"
            print(f"\n[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            # Return error response with details
            return JSONResponse({
                "error": error_msg,
                "uploaded": saved,
                "failed": f.filename,
                "details": file_details
            }, status_code=500)
    
    # All files processed successfully
    print(f"\n[OK] Upload complete! Returning details for {len(file_details)} files")
    for detail in file_details:
        print(f"   - {detail.get('type', 'unknown')}: {detail.get('filename', 'unknown')}")
    
    return JSONResponse({
        "uploaded": saved,
        "message": f"Successfully uploaded and indexed {len(saved)} file(s)",
        "details": file_details  # Include OCR text and other details
    })

@app.post("/query/")
async def query_endpoint(
    query: str = Form(None), 
    search_mode: str = Form("text"),  # "text", "image", or "audio"
    query_image: UploadFile = File(None),
    query_audio: UploadFile = File(None),
    top_k: int = Form(TOP_K_DEFAULT)
):
    global clip_embedder, ocr_engine  # Declare globals at function start
    
    try:
        print(f"\n{'='*60}")
        print(f"[SEARCH] Query received - Mode: {search_mode}")
        
        # Count chunks in both collections
        text_count = indexer.collection.count()
        clip_count = indexer.clip_collection.count()
        total_count = text_count + clip_count
        print(f"[STATS] Database has {total_count} total chunks ({text_count} text, {clip_count} CLIP)")
        
        # ALWAYS search BOTH collections to get all content including image OCR text
        search_k = max(top_k * 4, 15)  # Get more results to ensure good coverage
        
        # Determine query type and generate appropriate embeddings
        if search_mode == "image" and query_image:
            print("[IMAGE]  IMAGE SEARCH MODE: Using uploaded image as query")
            # Save image temporarily
            temp_path = os.path.join("data/uploads", f"temp_query_{uuid.uuid4().hex}_{query_image.filename}")
            os.makedirs("data/uploads", exist_ok=True)
            with open(temp_path, "wb") as out:
                content = await query_image.read()
                out.write(content)
            
            # Extract OCR text from query image
            if ocr_engine is None:
                from ocr_engine import get_ocr_engine
                ocr_engine = get_ocr_engine()
            
            # Open image for OCR
            from PIL import Image
            pil_img = Image.open(temp_path).convert("RGB")
            ocr_text = ocr_engine.extract_text(pil_img) if ocr_engine.is_available() else ""
            print(f"  [NOTE] OCR extracted: {len(ocr_text)} characters" if ocr_text else "  [NOTE] No OCR text detected")
            
            # FOLLOW TEXT QUERY WORKFLOW: Use OCR text as the query (like text mode uses user input)
            if ocr_text and len(ocr_text.strip()) > 10:
                # Fix specific known OCR spacing errors
                query = ocr_text.strip()
                
                # Fix common OCR spacing errors using simple replacement
                # These are the most common single-letter spacing issues
                spacing_fixes = {
                    ' j ob': ' job',
                    ' w ork': ' work',
                    ' s eeker': ' seeker',
                    ' s eekers': ' seekers',
                    ' e mploy': ' employ',
                    ' b enefit': ' benefit',
                    ' p otential': ' potential',
                    ' i mpact': ' impact',
                }
                
                for wrong, right in spacing_fixes.items():
                    query = query.replace(wrong, right)
                
                # Normalize multiple consecutive spaces to single space
                query = re.sub(r'\s{2,}', ' ', query)
                query = query.strip()
                print(f"  [NOTE] Using OCR text as query: '{query[:100]}...'")
            else:
                # No meaningful OCR - inform user
                query = ""
                print(f"  [WARNING]  No text detected in image. Please provide a text query or upload an image with text.")
            
            # EXACT SAME WORKFLOW AS TEXT MODE
            if query:
                print(f"  [SEARCH] Searching BOTH collections (text + CLIP) for comprehensive results...")
                
                # Search text collection (same as text mode)
                q_emb_text = embedder.embed_text(query)
                results_text = indexer.query(q_emb_text, n_results=search_k, use_clip=False)
                
                # Search CLIP collection (same as text mode)
                if clip_embedder is None:
                    print("  [LOAD] Loading CLIP model for image search...")
                    clip_embedder = get_clip_embedder(device=EMBED_DEVICE)
                
                q_emb_clip = clip_embedder.embed_text(query)
                results_clip = indexer.query(q_emb_clip, n_results=search_k, use_clip=True)
            else:
                # No query text available
                results_text = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
                results_clip = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
                
        elif search_mode == "audio" and query_audio:
            print("[AUDIO] AUDIO SEARCH MODE: Using uploaded audio as query")
            # Save audio temporarily
            temp_path = os.path.join("data/uploads", f"temp_query_{uuid.uuid4().hex}_{query_audio.filename}")
            os.makedirs("data/uploads", exist_ok=True)
            with open(temp_path, "wb") as out:
                content = await query_audio.read()
                out.write(content)
            
            # Convert WebM/MP4 to WAV if needed (for voice recording compatibility)
            converted_path = None
            file_ext = Path(temp_path).suffix.lower()
            if file_ext in ['.webm', '.mp4', '.m4a', '.ogg']:
                print(f"  [CONVERT] Converting {file_ext} to WAV for better compatibility...")
                try:
                    import subprocess
                    converted_path = temp_path.replace(file_ext, '.wav')
                    # Use FFmpeg to convert to WAV
                    subprocess.run([
                        'ffmpeg', '-i', temp_path, 
                        '-ar', '16000',  # 16kHz sample rate (Whisper standard)
                        '-ac', '1',       # Mono
                        '-y',             # Overwrite
                        converted_path
                    ], check=True, capture_output=True)
                    print(f"  [OK] Converted to WAV successfully")
                    # Use converted file for transcription
                    transcription_path = converted_path
                except Exception as conv_error:
                    print(f"  [WARNING]  Conversion failed: {conv_error}, trying original file...")
                    transcription_path = temp_path
            else:
                transcription_path = temp_path
            
            # Transcribe audio
            from ingestion import transcribe_audio_whisper
            res = transcribe_audio_whisper(transcription_path, model_name=WHISPER_MODEL, device=EMBED_DEVICE)
            query = res.get("text", "").strip()
            transcription_text = query  # Store for response
            print(f"  [NOTE] Transcribed audio query: {query[:100]}...")
            
            # EXACT SAME WORKFLOW AS TEXT MODE
            if query:
                print(f"  [SEARCH] Searching BOTH collections (text + CLIP) for comprehensive results...")
                
                # Search text collection (same as text mode)
                q_emb_text = embedder.embed_text(query)
                results_text = indexer.query(q_emb_text, n_results=search_k, use_clip=False)
                
                # Search CLIP collection (same as text mode)
                if clip_embedder is None:
                    print("  [LOAD] Loading CLIP model for image search...")
                    clip_embedder = get_clip_embedder(device=EMBED_DEVICE)
                
                q_emb_clip = clip_embedder.embed_text(query)
                results_clip = indexer.query(q_emb_clip, n_results=search_k, use_clip=True)
            else:
                # No transcription available
                results_text = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
                results_clip = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
                transcription_text = ""  # Empty transcription
            
            # Clean up temp files
            try:
                os.remove(temp_path)
                if converted_path and os.path.exists(converted_path):
                    os.remove(converted_path)
            except:
                pass
                
        else:  # Default: TEXT search mode
            print("[NOTE] TEXT SEARCH MODE: Using text query")
            if not query or not query.strip():
                return JSONResponse({"error": "Query text is required for text search mode"}, status_code=400)
            
            print(f"  Query: {query}")
            print("�[SEARCH] Searching BOTH collections (text + CLIP) for comprehensive results...")
            
            # Search text collection (documents, audio)
            q_emb_text = embedder.embed_text(query)
            results_text = indexer.query(q_emb_text, n_results=search_k, use_clip=False)
            
            # Search CLIP collection (images and their OCR text)
            if clip_embedder is None:
                print("  [LOAD] Loading CLIP model for image search...")
                clip_embedder = get_clip_embedder(device=EMBED_DEVICE)
            
            q_emb_clip = clip_embedder.embed_text(query)
            results_clip = indexer.query(q_emb_clip, n_results=search_k, use_clip=True)
        
        # Merge results from both collections
        retrieved_docs = (results_text.get("documents", [[]])[0] if results_text.get("documents") else []) + \
                        (results_clip.get("documents", [[]])[0] if results_clip.get("documents") else [])
        metadatas = (results_text.get("metadatas", [[]])[0] if results_text.get("metadatas") else []) + \
                   (results_clip.get("metadatas", [[]])[0] if results_clip.get("metadatas") else [])
        distances = (results_text.get("distances", [[]])[0] if results_text.get("distances") else []) + \
                   (results_clip.get("distances", [[]])[0] if results_clip.get("distances") else [])
        
        print(f"� Retrieved {len(retrieved_docs)} initial results ({len(results_text.get('documents', [[]])[0])} text, {len(results_clip.get('documents', [[]])[0])} CLIP)")
        
        # Analyze what we retrieved
        type_counts = {}
        embedding_type_counts = {}
        for meta in metadatas:
            file_type = meta.get("type", "unknown")
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
            embedding_type = meta.get("embedding_type", "standard")
            embedding_type_counts[embedding_type] = embedding_type_counts.get(embedding_type, 0) + 1
        print(f"📋 Result distribution: {type_counts}")
        print(f"[STATS] Embedding types: {embedding_type_counts}")
        
        # CRITICAL FIX: STRICTER THRESHOLDS + DOCUMENT PRIORITY
        # Problem: Irrelevant CLIP results (dist 0.75) beat relevant text (dist 1.1)
        # Solution: Much stricter CLIP thresholds + document quality bonus
        print("[TARGET] Ranking with strict quality thresholds + document priority")
        
        # Build combined results with metadata
        all_results = []
        for doc, meta, dist in zip(retrieved_docs, metadatas, distances):
            file_type = meta.get("type", "unknown")
            embedding_type = meta.get("embedding_type", "standard")
            all_results.append((doc, meta, dist, file_type, embedding_type))
        
        # CRITICAL FIX: Separate text and CLIP, apply STRICT quality standards
        text_results = []
        clip_results = []
        
        for doc, meta, dist, ftype, etype in all_results:
            if etype in ["clip_text", "clip_visual"]:
                clip_results.append((doc, meta, dist, ftype, etype))
            else:
                text_results.append((doc, meta, dist, ftype, etype))
        
        print(f"[STATS] Separated: {len(text_results)} text results, {len(clip_results)} CLIP results")
        
        # ABSOLUTE QUALITY THRESHOLDS (RESEARCH-GRADE STRICTNESS)
        # Text embeddings: More lenient (semantic text matching is reliable)
        TEXT_EXCELLENT_THRESHOLD = 0.8    # < 0.8 = excellent (lenient for PDFs)
        TEXT_GOOD_THRESHOLD = 1.2         # < 1.2 = good
        
        # CLIP embeddings: ULTRA STRICT (visual matching is unreliable for text queries)
        # Only truly relevant images should score high!
        CLIP_EXCELLENT_THRESHOLD = 0.55   # < 0.55 = excellent (ULTRA strict!)
        CLIP_GOOD_THRESHOLD = 0.70        # < 0.70 = good (very strict)
        
        # Apply absolute quality scoring with DOCUMENT PRIORITY BOOST
        scored_results = []
        
        for doc, meta, dist, ftype, etype in text_results:
            # Absolute quality score for text embeddings
            if dist < TEXT_EXCELLENT_THRESHOLD:
                quality_score = 1.0
                quality_label = "[STAR] EXCELLENT"
            elif dist < TEXT_GOOD_THRESHOLD:
                # Linear interpolation between thresholds
                ratio = (dist - TEXT_EXCELLENT_THRESHOLD) / (TEXT_GOOD_THRESHOLD - TEXT_EXCELLENT_THRESHOLD)
                quality_score = 1.0 - (0.2 * ratio)  # 1.0 to 0.8
                quality_label = "✓ GOOD"
            else:
                # Declining score after good threshold
                quality_score = max(0.0, 1.0 - (dist - TEXT_GOOD_THRESHOLD) / 0.8)
                quality_label = "~ FAIR" if quality_score > 0.3 else "✗ WEAK"
            
            # CRITICAL: Apply document priority boost (15% bonus for PDFs/DOCX)
            if ftype == "document":
                quality_score = min(1.0, quality_score * 1.15)  # 15% boost, cap at 1.0
                if quality_score >= 0.95:
                    quality_label = "[STAR][STAR] EXCELLENT (Document)"
            
            scored_results.append((doc, meta, dist, quality_score, ftype, etype, quality_label))
        
        for doc, meta, dist, ftype, etype in clip_results:
            # ULTRA STRICT quality score for CLIP embeddings
            # Only truly relevant images should score high!
            if dist < CLIP_EXCELLENT_THRESHOLD:
                quality_score = 1.0
                quality_label = "[STAR] EXCELLENT"
            elif dist < CLIP_GOOD_THRESHOLD:
                # Linear interpolation
                ratio = (dist - CLIP_EXCELLENT_THRESHOLD) / (CLIP_GOOD_THRESHOLD - CLIP_EXCELLENT_THRESHOLD)
                quality_score = 1.0 - (0.2 * ratio)  # 1.0 to 0.8
                quality_label = "✓ GOOD"
            else:
                # ULTRA STRICT: Declining score - CLIP results decay MUCH faster
                # After 0.70, quality drops off a cliff!
                quality_score = max(0.0, 1.0 - (dist - CLIP_GOOD_THRESHOLD) / 0.25)  # VERY fast decay
                quality_label = "~ FAIR" if quality_score > 0.4 else "✗ WEAK"
            
            scored_results.append((doc, meta, dist, quality_score, ftype, etype, quality_label))
        
        # Sort by quality score (document boost helps PDFs rank higher)
        scored_results.sort(key=lambda x: x[3], reverse=True)
        
        # Show top results with absolute quality scores
        print(f"[STATS] Top {min(10, len(scored_results))} results (absolute quality scoring):")
        for i, (doc, meta, dist, quality_score, ftype, etype, quality_label) in enumerate(scored_results[:10]):
            emoji = "[DOC]" if ftype == "document" else "[IMAGE]" if ftype == "image" else "[AUDIO]" if ftype == "audio" else "❓"
            source_name = meta.get('source', 'unknown')[:40]
            quality_pct = quality_score * 100
            
            print(f"  [{i+1}] {emoji} {ftype} - {source_name}")
            print(f"      Quality: {quality_pct:.1f}% {quality_label} | Type: {etype}")
            print(f"      Raw distance: {dist:.4f}")
        
        # CRITICAL: Filter by STRICT quality threshold (research-grade)
        # Only use results with strong semantic match
        QUALITY_THRESHOLD = 0.70  # Minimum 70% quality (STRICT for national org)
        
        print(f"\n[TARGET] Applying STRICT quality threshold: {QUALITY_THRESHOLD*100:.0f}% (research-grade)")
        filtered_results = [r for r in scored_results if r[3] >= QUALITY_THRESHOLD]
        
        if len(filtered_results) < len(scored_results):
            removed_count = len(scored_results) - len(filtered_results)
            print(f"   [ERROR] Filtered out {removed_count} low-quality results")
        
        if not filtered_results:
            print("   [WARNING]  WARNING: No results meet strict threshold! Lowering to 60%...")
            QUALITY_THRESHOLD = 0.60
            filtered_results = [r for r in scored_results if r[3] >= QUALITY_THRESHOLD]
            
        if not filtered_results:
            print("   [WARNING]  Still no results! Using top 3 anyway (check data quality)")
            filtered_results = scored_results[:3] if scored_results else []
        
        # Take top_k from filtered results
        final_results_with_scores = filtered_results[:top_k]
        
        # Remove duplicates while preserving order and quality scores
        seen_docs = set()
        unique_results = []
        for doc, meta, dist, quality_score, ftype, etype, quality_label in final_results_with_scores:
            doc_preview = doc[:100] if doc else ""
            if doc_preview not in seen_docs:
                seen_docs.add(doc_preview)
                unique_results.append((doc, meta, dist, quality_score, ftype, etype, quality_label))
        
        # Limit to top_k
        unique_results = unique_results[:top_k]
        
        # Unpack final results
        retrieved_docs = [item[0] for item in unique_results]
        metadatas = [item[1] for item in unique_results]
        distances = [item[2] for item in unique_results]
        quality_scores = [item[3] for item in unique_results]  # Keep quality scores
        
        print(f"\n[OK] Using {len(retrieved_docs)} most relevant documents for answer generation")
        
        # Show final selected documents with clear quality scores
        print(f"📋 Final selections:")
        for i, (doc, meta, quality_score, dist) in enumerate(zip(retrieved_docs[:5], metadatas[:5], quality_scores[:5], distances[:5])):
            file_type = meta.get('type', 'unknown')
            emoji = "[DOC]" if file_type == "document" else "[IMAGE]" if file_type == "image" else "[AUDIO]" if file_type == "audio" else "❓"
            quality_pct = quality_score * 100
            print(f"  [{i+1}] {emoji} {file_type}: {meta.get('source', 'unknown')[:40]}")
            print(f"      Quality: {quality_pct:.1f}% | Distance: {dist:.4f} | Preview: {doc[:60]}...")
        
        # Build prompt
        prompt = build_prompt(query, retrieved_docs, metadatas, top_k)
        print(f"[NOTE] Prompt built, length: {len(prompt)} characters")
        if not retrieved_docs:
            print("[WARNING]  WARNING: No documents retrieved! Answer may be generic.")
        
        # Generate answer - optimized to 800 tokens for faster, focused responses
        raw_answer = generate_answer(prompt, model_path=LLAMA_MODEL_PATH, max_tokens=800, temperature=0.3, device=EMBED_DEVICE)
        print(f"Raw answer length: {len(raw_answer)} characters")
        print(f"Raw answer preview: {raw_answer[:200]}...")
        
        # POST-PROCESS: Force proper formatting (ChatGPT-style structure)
        formatted_sources = [{"source": m.get("source", "Unknown"), "type": m.get("type", "document")} for m in metadatas]
        answer = format_with_sources(raw_answer, formatted_sources)
        print(f"[OK] Formatted answer length: {len(answer)} characters")
        print(f"{'='*60}\n")
        
        # Enrich metadata for better citation display with quality scores
        enriched_sources = []
        for i, (doc, meta, quality_score) in enumerate(zip(retrieved_docs, metadatas, quality_scores)):
            # Get file path and ensure it's accessible
            file_path = meta.get("path", "")
            if file_path and not os.path.isabs(file_path):
                file_path = os.path.abspath(file_path)
            
            # Use quality score (0-1 scale) for display
            source_info = {
                "citation_number": i + 1,
                "document": doc,
                "metadata": meta,
                "similarity_score": float(quality_score),  # Use quality score (0-1)
                "relevance_percent": float(quality_score * 100),  # Also provide as percentage
                "source_file": meta.get("source", "Unknown"),
                "file_type": meta.get("type", "unknown"),
                "file_path": file_path,  # Use absolute path
            }
            # Add type-specific information
            if meta.get("type") == "audio":
                source_info["start_time"] = meta.get("start", 0)
                source_info["end_time"] = meta.get("end", 0)
                source_info["duration"] = meta.get("end", 0) - meta.get("start", 0)
            elif meta.get("type") == "document":
                source_info["chunk_index"] = meta.get("chunk", 0)
                source_info["char_start"] = meta.get("start", 0)
                source_info["char_end"] = meta.get("end", 0)
            elif meta.get("type") == "image":
                source_info["ocr_text"] = meta.get("ocr_text", "")
            
            enriched_sources.append(source_info)
        
        # Calculate answer confidence score
        citation_count = len(re.findall(r'\[\d+\]', answer))
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        source_coverage = min(citation_count / max(len(enriched_sources), 1), 1.0)
        
        # Confidence formula: 40% average quality + 30% citation density + 30% source coverage
        confidence_score = (avg_quality * 0.4) + (min(citation_count / 10, 1.0) * 0.3) + (source_coverage * 0.3)
        confidence_percent = round(confidence_score * 100, 1)
        
        # Determine confidence level
        if confidence_percent >= 80:
            confidence_level = "High"
        elif confidence_percent >= 60:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Build response
        response_data = {
            "answer": answer, 
            "sources": enriched_sources,
            "query": query,
            "num_sources": len(enriched_sources),
            "confidence": {
                "score": confidence_percent,
                "level": confidence_level,
                "citation_count": citation_count,
                "avg_source_quality": round(avg_quality * 100, 1)
            }
        }
        
        # Add transcription if audio mode was used
        if search_mode == "audio" and 'transcription_text' in locals():
            response_data["transcription"] = transcription_text
        
        return JSONResponse(response_data)
    except Exception as e:
        print(f"Error in query endpoint: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"answer": f"Error: {str(e)}", "sources": [], "query": query}, status_code=500)

@app.get("/view_source/{file_path:path}")
async def view_source(file_path: str):
    """View or download a source file"""
    try:
        from urllib.parse import unquote
        
        # Decode URL encoding (handles %20 for spaces, etc.)
        decoded_path = unquote(file_path)
        print(f"\n[SEARCH] View source request:")
        print(f"   Original: {file_path}")
        print(f"   Decoded: {decoded_path}")
        
        # Normalize path separators (handle Windows backslashes)
        normalized_path = decoded_path.replace('\\', os.sep).replace('/', os.sep)
        print(f"   Normalized: {normalized_path}")
        
        # Try different path resolutions
        paths_to_try = [
            normalized_path,  # As provided
            os.path.abspath(normalized_path),  # Absolute
            os.path.join(os.getcwd(), normalized_path),  # Relative to CWD
        ]
        
        # If path doesn't include 'uploads', try adding it
        if 'uploads' not in normalized_path.lower():
            filename = os.path.basename(normalized_path)
            paths_to_try.append(os.path.join("data/uploads", filename))
        
        # Try each path
        full_path = None
        for attempt_path in paths_to_try:
            print(f"   Trying: {attempt_path}")
            if os.path.exists(attempt_path):
                full_path = attempt_path
                print(f"   [OK] Found!")
                break
        
        if full_path is None:
            print(f"   [ERROR] File not found in any location")
            
            # List similar files in uploads for debugging
            uploads_dir = "data/uploads"
            if os.path.exists(uploads_dir):
                all_files = os.listdir(uploads_dir)
                # Find files with similar names
                basename = os.path.basename(normalized_path)
                similar = [f for f in all_files if basename[:20] in f or f[:20] in basename]
                print(f"   Similar files in uploads: {similar[:3]}")
            
            return JSONResponse({
                "error": f"File not found: {decoded_path}",
                "searched_paths": paths_to_try,
                "suggestion": "File may have been deleted or moved"
            }, status_code=404)
        
        print(f"   📁 Serving file: {full_path}")
        return FileResponse(full_path)
        
    except Exception as e:
        print(f"   [ERROR] Error serving file: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/stats/")
async def get_stats():
    """Get statistics about indexed documents"""
    try:
        # Get all documents from both collections
        text_data = indexer.collection.get()
        clip_data = indexer.clip_collection.get()
        
        text_metadatas = text_data.get("metadatas", [])
        clip_metadatas = clip_data.get("metadatas", [])
        all_metadatas = text_metadatas + clip_metadatas
        
        stats = {
            "total_chunks": len(all_metadatas),
            "text_collection_chunks": len(text_metadatas),
            "clip_collection_chunks": len(clip_metadatas),
            "by_type": {},
            "by_source": {}
        }
        
        for meta in all_metadatas:
            if not isinstance(meta, dict):
                continue
            doc_type = meta.get("type", "unknown")
            source = meta.get("source", "unknown")
            
            stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
        
        return JSONResponse(stats)
    except Exception as e:
        print(f"Error in stats endpoint: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e), "total_chunks": 0, "by_type": {}, "by_source": {}}, status_code=200)

@app.get("/list_files/")
async def list_indexed_files():
    """List all indexed files with their metadata"""
    try:
        all_data = indexer.collection.get()
        metadatas = all_data.get("metadatas", [])
        
        # Group by source file
        files_dict = {}
        for meta in metadatas:
            if not isinstance(meta, dict):
                continue
            source = meta.get("source", "unknown")
            if source not in files_dict:
                files_dict[source] = {
                    "source": source,
                    "path": meta.get("path", ""),
                    "type": meta.get("type", "unknown"),
                    "chunks": 0
                }
            files_dict[source]["chunks"] += 1
        
        return JSONResponse({"files": list(files_dict.values())})
    except Exception as e:
        print(f"Error in list_files endpoint: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e), "files": []}, status_code=200)

@app.get("/source_metadata/{file_path:path}")
async def get_source_metadata(file_path: str):
    """Get detailed metadata for a source file"""
    try:
        from urllib.parse import unquote
        import datetime
        from PIL import Image
        
        decoded_path = unquote(file_path)
        normalized_path = decoded_path.replace('\\', os.sep).replace('/', os.sep)
        
        # Try to find the file
        full_path = None
        if os.path.exists(normalized_path):
            full_path = normalized_path
        elif os.path.exists(os.path.abspath(normalized_path)):
            full_path = os.path.abspath(normalized_path)
        
        if not full_path or not os.path.exists(full_path):
            return JSONResponse({"error": "File not found"}, status_code=404)
        
        # Get basic file stats
        stats = os.stat(full_path)
        file_size = stats.st_size
        modified_time = datetime.datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        created_time = datetime.datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        
        # Determine file type
        extension = os.path.splitext(full_path)[1].lower()
        
        metadata = {
            "file_name": os.path.basename(full_path),
            "file_path": full_path,
            "file_size": file_size,
            "file_size_formatted": f"{file_size / 1024:.2f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB",
            "modified_date": modified_time,
            "created_date": created_time,
            "extension": extension
        }
        
        # Type-specific metadata
        if extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            try:
                img = Image.open(full_path)
                metadata["type"] = "image"
                metadata["dimensions"] = f"{img.width} x {img.height}"
                metadata["width"] = img.width
                metadata["height"] = img.height
                metadata["format"] = img.format
                metadata["mode"] = img.mode
                
                # Try to get EXIF data
                exif_data = img._getexif() if hasattr(img, '_getexif') else None
                if exif_data:
                    metadata["exif"] = {str(k): str(v) for k, v in exif_data.items() if k and v}
            except Exception as e:
                metadata["error"] = f"Could not read image metadata: {str(e)}"
        
        elif extension in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
            metadata["type"] = "audio"
            # Try to get audio duration if possible
            try:
                import wave
                if extension == '.wav':
                    with wave.open(full_path, 'r') as audio:
                        frames = audio.getnframes()
                        rate = audio.getframerate()
                        duration = frames / float(rate)
                        metadata["duration"] = f"{duration:.2f} seconds"
                        metadata["sample_rate"] = rate
                        metadata["channels"] = audio.getnchannels()
            except:
                pass
        
        elif extension in ['.pdf', '.txt', '.docx', '.doc']:
            metadata["type"] = "document"
            if extension == '.txt':
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        metadata["word_count"] = len(content.split())
                        metadata["char_count"] = len(content)
                        metadata["line_count"] = len(content.split('\n'))
                except:
                    pass
        
        # Get chunks from ChromaDB for this file
        try:
            all_data = indexer.collection.get()
            metadatas = all_data.get("metadatas", [])
            source_name = os.path.basename(full_path)
            
            chunks_for_file = [m for m in metadatas if m.get("source") == source_name]
            metadata["chunk_count"] = len(chunks_for_file)
            
            # Get OCR confidence if available
            ocr_confidences = [m.get("ocr_confidence") for m in chunks_for_file if m.get("ocr_confidence")]
            if ocr_confidences:
                metadata["avg_ocr_confidence"] = f"{sum(ocr_confidences) / len(ocr_confidences):.1f}%"
        except:
            pass
        
        return JSONResponse(metadata)
        
    except Exception as e:
        print(f"Error in source_metadata endpoint: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/query_by_image/")
async def query_by_image(image: UploadFile = File(...), top_k: int = Form(TOP_K_DEFAULT)):
    """Search using an image (image-to-text/image cross-modal search)"""
    try:
        print(f"Image query received: {image.filename}")
        
        # Save image temporarily
        temp_path = os.path.join("data/uploads", f"temp_query_{uuid.uuid4().hex}_{image.filename}")
        os.makedirs("data/uploads", exist_ok=True)
        with open(temp_path, "wb") as out:
            content = await image.read()
            out.write(content)
        
        # Load and embed image
        pil_img = load_image(temp_path)
        img_emb = embedder.embed_image(pil_img)
        print(f"Image embedding shape: {img_emb.shape}")
        
        # Also extract OCR text for context
        global ocr_engine
        if ocr_engine is None:
            ocr_engine = get_ocr_engine()
        ocr_text = ocr_engine.extract_text(pil_img) if ocr_engine.is_available() else ""
        print(f"OCR extracted: {ocr_text[:100]}..." if ocr_text else "No OCR text")
        
        # Query using image embedding
        results = indexer.query(img_emb, n_results=top_k)
        retrieved_docs = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        print(f"Retrieved {len(retrieved_docs)} documents for image query")
        
        # Build context for LLM
        query_text = f"Based on this image" + (f" which contains text: '{ocr_text}'" if ocr_text else "") + ", what relevant information can you find?"
        prompt = build_prompt(query_text, retrieved_docs, metadatas, top_k)
        
        # Generate answer - increased max_tokens to 2048 for complete, well-formatted answers with ALL sections
        raw_answer = generate_answer(prompt, model_path=LLAMA_MODEL_PATH, max_tokens=800, temperature=0.3, device=EMBED_DEVICE)
        
        # POST-PROCESS: Force proper formatting (ChatGPT-style structure)
        formatted_sources = [{"source": m.get("source", "Unknown"), "type": m.get("type", "document")} for m in metadatas]
        answer = format_with_sources(raw_answer, formatted_sources)
        print(f"[OK] Image query formatted answer length: {len(answer)} characters")
        
        # Enrich metadata
        enriched_sources = []
        for i, (doc, meta, dist) in enumerate(zip(retrieved_docs, metadatas, distances)):
            source_info = {
                "citation_number": i + 1,
                "document": doc,
                "metadata": meta,
                "similarity_score": float(1 - dist) if dist else 1.0,
                "source_file": meta.get("source", "Unknown"),
                "file_type": meta.get("type", "unknown"),
                "file_path": meta.get("path", ""),
            }
            if meta.get("type") == "audio":
                source_info["start_time"] = meta.get("start", 0)
                source_info["end_time"] = meta.get("end", 0)
                source_info["duration"] = meta.get("end", 0) - meta.get("start", 0)
            elif meta.get("type") == "document":
                source_info["chunk_index"] = meta.get("chunk", 0)
            elif meta.get("type") == "image":
                source_info["ocr_text"] = meta.get("ocr_text", "")
            enriched_sources.append(source_info)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return JSONResponse({
            "answer": answer,
            "sources": enriched_sources,
            "query": query_text,
            "ocr_text": ocr_text,
            "num_sources": len(enriched_sources)
        })
    except Exception as e:
        print(f"Error in image query endpoint: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"answer": f"Error: {str(e)}", "sources": []}, status_code=500)


