import re

def extract_sections(text: str):
    """
    Bypasses unpredictable SEC HTML header variations by treating 
    the text as a single unified narrative body optimized for sliding windows.
    """
    return [{
        "item": "10K_BODY",
        "title": "Filing Main Content",
        "text": text.strip()
    }]

def chunk_section(section, year, source_file, chunk_size=800, overlap=150, global_offset=0):
    """
    Slices the filing body using strict character lengths to prevent token cap exceptions
    and guarantee consistent chunk volume density across all historical files.
    """
    text_content = section["text"]
    chunks = []
    
    # Standardize token safety boundaries (1 char roughly maps to 0.25 tokens)
    # 8,000 characters provides a stable, safe margin under the model's 8,192 token limit.
    max_char_limit = 8000 
    char_overlap = 1200
    char_step = max_char_limit - char_overlap
    
    if not text_content.strip():
        return chunks

    for c_start in range(0, len(text_content), char_step):
        sub_text = text_content[c_start:c_start + max_char_limit].strip()
        if len(sub_text) < 200: # Ignore tiny final trailing lines
            continue
            
        unique_index = global_offset + len(chunks)
        chunk_id = f"{year}_CHUNK_{unique_index}"
        
        chunks.append({
            "chunk_id": chunk_id,
            "year": int(year),
            "item": "10K_BODY",
            "section": "Filing Main Content",
            "source_file": source_file,
            "text": sub_text,
        })
        
    return chunks
