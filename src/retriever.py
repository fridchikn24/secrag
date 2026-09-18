import json
import re
from pathlib import Path
import faiss
import numpy as np
from src.embeddings import embed_texts

INDEX_DIR = Path("indexes")

def infer_target_year_from_query(query_text: str) -> int:
    q_lower = query_text.lower()
    explicit_years = [int(y) for y in re.findall(r'\b(20\d{2})\b', query_text)]
    if explicit_years:
        return explicit_years[0]
    if "recent" in q_lower or "latest" in q_lower or "current" in q_lower:
        return 2025
    return None

def retrieve_from_partition(query_vector, year: int, k: int = 5):
    """Safely reads and queries an isolated index partition."""
    idx_path = INDEX_DIR / f"faiss_{year}.index"
    meta_path = INDEX_DIR / f"metadata_{year}.json"
    
    if not idx_path.exists() or not meta_path.exists():
        return []
        
    index = faiss.read_index(str(idx_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    distances, indices = index.search(query_vector, min(k, len(chunks)))
    retrieved = []
    for rank_idx in indices.ravel():
        if rank_idx == -1 or rank_idx >= len(chunks): continue
        retrieved.append(chunks[rank_idx])
    return retrieved[:k]

def retrieve(query_text: str, k: int = 5):
    query_vector = np.asarray(embed_texts([query_text]), dtype="float32")
    faiss.normalize_L2(query_vector)
    
    target_year = infer_target_year_from_query(query_text)
    is_comparative = any(w in query_text.lower() for w in ["change", "between", "across", "compare"])
    
    retrieved_sources = []
    
    #  BALANCED ROUTING ROUTE:
    if is_comparative and target_year == 2025:
        # Pull evenly from the two most recent distinct database files
        retrieved_sources.extend(retrieve_from_partition(query_vector, 2024, k=3))
        retrieved_sources.extend(retrieve_from_partition(query_vector, 2025, k=3))
    elif target_year:
        # Strict context filtering path
        retrieved_sources.extend(retrieve_from_partition(query_vector, target_year, k=k))
    else:
        # Global fallback across all index files if query has no specific year criteria
        for year in [2021, 2022, 2023, 2024, 2025]:
            retrieved_sources.extend(retrieve_from_partition(query_vector, year, k=2))
            
    return retrieved_sources[:k]
