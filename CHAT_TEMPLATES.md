# Chat Template Configuration

## Overview

This project uses **chat templates** to format prompts for different LLM models. The chat template is stored in `backend/utils/utils.py` in the `build_prompt()` function.

## Current Configuration

**Active Model**: Mistral 7B  
**Template Format**: `[INST]...[/INST]`  
**Location**: `backend/utils/utils.py` (lines 6-120)

## Chat Template Formats by Model

### Mistral 7B (Current)
```python
prompt = f"""[INST] {system_instructions}

{context}

Question: {question} [/INST]

{assistant_start}"""
```

### Phi-3 (Alternative)
```python
prompt = f"""<|system|>
{system_instructions}<|end|>
<|user|>
{context}
{question}<|end|>
<|assistant|>
{assistant_start}"""
```

### LLaMA 2 (Alternative)
```python
prompt = f"""[INST] <<SYS>>
{system_instructions}
<</SYS>>

{context}

{question} [/INST]
{assistant_start}"""
```

### LLaMA 3 (Alternative)
```python
prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_instructions}<|eot_id|><|start_header_id|>user<|end_header_id|>

{context}
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_start}"""
```

## How to Change Models

### 1. Update .env File
```env
# Change this to your desired model
LLAMA_MODEL_PATH=mistral  # or phi3, llama2, llama3, etc.
```

### 2. Pull the Model via Ollama
```bash
# For Mistral
ollama pull mistral

# For Phi-3
ollama pull phi3

# For LLaMA 2
ollama pull llama2

# For LLaMA 3
ollama pull llama3
```

### 3. Update Chat Template (if needed)

**File**: `backend/utils/utils.py`  
**Function**: `build_prompt()`  
**Lines**: 6-120

If switching to a different model family, update the template format:

```python
def build_prompt(question: str, retrieved_docs: List[str], metadatas: List[Dict], top_k: int = 5) -> str:
    # ... context building code ...
    
    # For Mistral (current):
    prompt = f"""[INST] {instructions}
{context}
Question: {question} [/INST]

## Summary
"""
    
    # OR for Phi-3:
    # prompt = f"""<|system|>
    # {instructions}<|end|>
    # <|user|>
    # {context}
    # {question}<|end|>
    # <|assistant|>
    # ## Summary
    # """
    
    return prompt
```

## Model Detection

The system automatically detects which model you're using in `backend/models/generator.py`:

```python
def _try_load_ollama(self):
    model_name = self.gptq_path  # From .env LLAMA_MODEL_PATH
    
    # Auto-detect model type
    model_lower = str(model_name).lower()
    if "mistral" in model_lower:
        model_name = "mistral"
    elif "phi-3" in model_lower or "phi3" in model_lower:
        model_name = "phi3"
    elif "llama" in model_lower:
        model_name = "llama2"  # Adjust based on version
```

## Template Components

### 1. System Instructions (lines 80-115 in utils.py)
```python
system_instructions = """You are a professional AI assistant that generates ChatGPT-style answers.

FORMATTING RULES:
- Use headers (##), bold (**text**), bullets
- Blank line after headers
- Bullet points under 20 words
...

CITATION REQUIREMENTS:
- Add numbered citations [1], [2], [3] after EVERY factual claim
- Use [1] for information from Source 1, [2] for Source 2, etc.
...
"""
```

### 2. Context Sources (lines 30-70 in utils.py)
```python
context = """[Source 1] DOCUMENT: report.pdf (page 3)
Content: The market grew by 15% in Q4...

[Source 2] TEXT FROM IMAGE: chart.png (OCR Extracted)
Content: Revenue reached $2.5M...

[Source 3] AUDIO: meeting.mp3 (Transcription, at 45-67s)
Content: Our strategy focuses on...
"""
```

### 3. User Question
```python
question = "What was the revenue growth?"
```

### 4. Assistant Start
```python
assistant_start = """## Summary

"""
```

## Testing Your Template

After changing the template, test it:

```bash
# Start the server
python run_server.py

# Test via web UI at http://localhost:8000
# Or test via API:
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the summary?"}'
```

## Common Issues

### Issue 1: Model Not Generating Responses
**Cause**: Wrong chat template format for your model  
**Solution**: Update `build_prompt()` in `backend/utils/utils.py` to match your model's format

### Issue 2: Incomplete Responses
**Cause**: Template end token not recognized  
**Solution**: Ensure you're using the correct end tokens (`[/INST]`, `<|end|>`, etc.)

### Issue 3: Model Not Found
**Cause**: Ollama model not pulled or wrong name in .env  
**Solution**: Run `ollama pull <model_name>` and update `LLAMA_MODEL_PATH` in `.env`

## Advanced: Multi-Model Support

To support multiple models simultaneously, add model detection in `build_prompt()`:

```python
def build_prompt(question: str, retrieved_docs: List[str], metadatas: List[Dict], 
                 model_name: str = "mistral", top_k: int = 5) -> str:
    # ... context building ...
    
    # Auto-detect template based on model
    if "mistral" in model_name.lower():
        template_start = "[INST]"
        template_end = "[/INST]"
    elif "phi" in model_name.lower():
        template_start = "<|system|>"
        template_end = "<|assistant|>"
    elif "llama3" in model_name.lower():
        template_start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        template_end = "<|start_header_id|>assistant<|end_header_id|>"
    else:
        # Default to Mistral format
        template_start = "[INST]"
        template_end = "[/INST]"
    
    prompt = f"""{template_start} {instructions}
{context}
Question: {question} {template_end}

## Summary
"""
    return prompt
```

## Summary

- **Chat template location**: `backend/utils/utils.py`, `build_prompt()` function
- **Current format**: Mistral `[INST]...[/INST]`
- **Model selection**: Set `LLAMA_MODEL_PATH` in `.env`
- **Model loading**: `backend/models/generator.py`, `_try_load_ollama()` function
- **To change**: Update `.env`, pull model via Ollama, update template if needed

For more information about model management, see [MODELS.md](MODELS.md).
