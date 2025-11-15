# utils.py
from typing import List, Dict

def build_prompt(question: str, retrieved_docs: List[str], metadatas: List[Dict], top_k: int = 5) -> str:
    if not retrieved_docs:
        return f"""[INST] You are a professional AI assistant.

FORMATTING RULES:
- Use headers (##), bold (**text**), bullets
- Blank line after headers
- Bullet points under 20 words
- Paragraphs 2-3 sentences max
- Friendly ChatGPT-like tone

STRUCTURE:
1. ## Summary (1-2 lines)
2. ## Key Points (3-5 bullets)
3. ## Steps (if applicable)
4. ## Insights (short paragraphs)
5. ## Takeaway (one sentence)

Since no sources are available, provide a general answer.

Question: {question} [/INST]

## Summary

"""
    
    max_docs = min(8, len(retrieved_docs))
    max_chars_per_doc = 1000
    
    context_lines = []
    has_ocr_text = False
    for i, (doc, md) in enumerate(zip(retrieved_docs[:max_docs], metadatas[:max_docs])):
        src = md.get("source", "unknown")
        doc_type = md.get("type", "document")
        embedding_type = md.get("embedding_type", "standard")
        
        if doc_type == "image":
            type_label = "IMAGE"
            extra = []
            if embedding_type == "clip_text" or md.get("ocr_text"):
                type_label = "TEXT FROM IMAGE"
                extra.append("OCR Extracted")
                has_ocr_text = True
            elif embedding_type == "clip_visual":
                extra.append("Visual Content")
        elif doc_type == "audio":
            type_label = "AUDIO"
            extra = ["Transcription"]
        elif doc_type == "document":
            type_label = "DOCUMENT"
            extra = []
        else:
            type_label = f"{doc_type.upper()}"
            extra = []
        
        if "page" in md:
            extra.append(f"page {md['page']}")
        if "start" in md and "end" in md:
            extra.append(f"at {md['start']:.0f}-{md['end']:.0f}s")
        if "chunk" in md:
            extra.append(f"section {md['chunk']}")
        
        extra_txt = (" (" + ", ".join(extra) + ")") if extra else ""
        truncated_doc = doc[:max_chars_per_doc] + ("..." if len(doc) > max_chars_per_doc else "")
        
        context_lines.append(f"[Source {i+1}] {type_label}: {src}{extra_txt}\nContent: {truncated_doc}")
    
    context = "\n\n".join(context_lines)
    
    ocr_emphasis = ""
    if has_ocr_text:
        ocr_emphasis = "\n\nIMPORTANT: Some sources contain TEXT EXTRACTED FROM IMAGES (OCR). Treat them as primary text sources."
    
    prompt = f"""[INST] You are a professional AI assistant that generates ChatGPT-style answers.

FORMATTING RULES:
- Use headers (##), bold (**text**), bullets
- Blank line after headers
- Bullet points under 20 words
- Paragraphs 2-3 sentences max
- Friendly tone

STRUCTURE:
1. ## Summary (1-2 lines)
2. ## Key Points (3-5 bullets under 20 words)
3. ## Steps (if applicable)
4. ## Insights (short paragraphs)
5. ## Takeaway (one sentence)

CRITICAL CITATION REQUIREMENTS:
- Add numbered citations [1], [2], [3] after EVERY factual claim
- Use [1] for information from Source 1, [2] for Source 2, etc.
- Multiple sources? Use [1][2] together
- Citations MUST appear inline within sentences
- Each paragraph must have at least one citation

CITATION EXAMPLES:
- "Job seekers face increased competition [1]."
- "AI tools help automate tasks [2][3]."
- "The market shifted in 2024 [1]."

REMEMBER:
- Answer ONLY based on the provided Context sources
- Do NOT make up information not in the sources
- Every fact needs a citation [1], [2], etc.
- If sources don't have the answer, say so clearly

Context:
{context}{ocr_emphasis}

Question: {question}

IMPORTANT: Follow structure: Summary -> Key Points -> Steps -> Insights -> Takeaway. 
Add [1], [2], [3] citations after EVERY fact. Use [Source X] number. Bullets under 20 words. Bold key terms. [/INST]

## Summary

"""
    
    return prompt
