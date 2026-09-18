from dotenv import load_dotenv
load_dotenv()

from src.rag import SECRAG

rag = SECRAG()

while True:
    question = input("\nQuestion (or 'exit'): ").strip()

    if question.lower() == "exit":
        break

    result = rag.answer(question, k=5)

    print("\nANSWER\n------")
    print(result["answer"])

    print("\nSOURCES\n-------")
    for source in result["sources"]:
        
        section_title = source.get("section") or source.get("title") or "Unknown Section"
        print(
            f"{source.get('chunk_id', 'N/A')} | {source.get('year', 'N/A')} | "
            f"{section_title} | {source.get('source_file', 'N/A')}"
        )
