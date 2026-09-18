import json
import re
import shutil
from pathlib import Path
import faiss
import numpy as np
from dotenv import load_dotenv

from src.parser import parse_filing
from src.chunker import extract_sections, chunk_section
from src.embeddings import embed_texts

load_dotenv()

FILINGS_DIR = Path("data/filings")
INDEX_DIR = Path("indexes")

if INDEX_DIR.exists():
    shutil.rmtree(INDEX_DIR)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Scan for subdirectories representing each year
year_dirs = sorted([d for d in FILINGS_DIR.iterdir() if d.is_dir() and d.name.isdigit()])
print(f" Found {len(year_dirs)} year directories: {[d.name for d in year_dirs]}")

for y_dir in year_dirs:
    year = int(y_dir.name)
    filing_paths = sorted([p for p in y_dir.iterdir() if p.is_file() and p.suffix.lower() in {".html", ".htm"}])
    
    if not filing_paths:
        continue
        
    print(f"\n Building Isolated Index Partition for Year: {year}")
    year_chunks = []
    
    for path in filing_paths:
        file_text = str(parse_filing(str(path)))
        if len(file_text.strip()) == 0:
            continue
            
        sections = extract_sections(file_text)
        for section in sections:
            chunks = chunk_section(
                section=section,
                year=year,
                source_file=path.name,
                global_offset=len(year_chunks)
            )
            year_chunks.extend(chunks)

    if not year_chunks:
        print(f" No chunks generated for year {year}. Skipping.")
        continue

    print(f" Embedding {len(year_chunks)} chunks for {year} partition...")
    vectors = np.asarray(embed_texts([c["text"] for c in year_chunks]), dtype="float32")
    faiss.normalize_L2(vectors)

    # Save isolated assets explicitly per year
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_DIR / f"faiss_{year}.index"))

    with open(INDEX_DIR / f"metadata_{year}.json", "w", encoding="utf-8") as f:
        json.dump(year_chunks, f, indent=2)

print("\n Success! Multi-index partitioning successfully written to disk.")
