
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from pathlib import Path
import re



import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from pathlib import Path
import re

def parse_html(path: str) -> str:
    html_content = Path(path).read_text(encoding="utf-8", errors="ignore")
    
    # Suppress verbose parsing alerts from SEC structural layouts
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    
    # Check if the file is packed with custom 'ix:' markup tags
    if "ix:" in html_content:
        # Match any text packed between open/close tags or attributes
        # Strips out formatting properties while keeping clean numeric/narrative metrics
        clean_text = re.sub(r'<[^>]+>', ' ', html_content)
        combined_text = re.sub(r'\s+', ' ', clean_text).strip()
    else:
        # --- Fallback Option for Standard Legacy HTML Layouts (like 2021) ---
        soup = BeautifulSoup(html_content, "lxml")
        
        for tag in soup(["script", "style", "noscript", "head", "metadata"]):
            tag.decompose()
            
        parsed_blocks = []
        for element in soup.find_all(['p', 'table', 'h1', 'h2', 'h3', 'h4', 'div']):
            if any(parent.name in ['p', 'table', 'h1', 'h2', 'h3', 'h4'] for parent in element.parents):
                continue
                
            if element.name == 'table':
                markdown_table = []
                for row in element.find_all('tr'):
                    cells = [re.sub(r'\s+', ' ', cell.get_text(" ", strip=True)) for cell in row.find_all(['td', 'th'])]
                    if any(cells):
                        markdown_table.append("| " + " | ".join(cells) + " |")
                
                if markdown_table and len(markdown_table) > 1:
                    col_count = markdown_table.count('|') - 1
                    markdown_table.insert(1, "|" + "---| " * col_count)
                    parsed_blocks.append("\n".join(markdown_table))
            else:
                text = element.get_text(" ", strip=True)
                text = re.sub(r'\s+', ' ', text)
                if len(text) > 5:
                    parsed_blocks.append(text)
                    
        combined_text = "\n\n".join(parsed_blocks)

    words = combined_text.split()
    sanitized_words = []
    
    for word in words:
        # Exclude raw XBRL database URLs that pollute vector weights
        if "http" in word or "fasb.org" in word or "us-gaap" in word or "xbrl" in word:
            continue
        # Split any glitched long string patterns seamlessly
        if len(word) > 100:
            sanitized_words.extend([word[i:i+15] for i in range(0, len(word), 15)])
        else:
            sanitized_words.append(word)
            
    return " ".join(sanitized_words)



def parse_filing(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".html", ".htm"}:
        return parse_html(path)
    raise ValueError(f"Unsupported filing format: {suffix}")
