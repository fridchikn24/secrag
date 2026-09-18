from openai import OpenAI

SYSTEM_PROMPT = """You are a financial research assistant answering questions about SEC 10-K filings.

Rules:
1. Use only the supplied evidence.
2. Do not invent facts, numbers, dates, or explanations.
3. If the evidence does not answer the question, say that the filings do not provide enough information.
4. Preserve numerical precision and distinguish millions from billions.
5. Cite every material claim using the exact supplied SOURCE_ID (e.g., [SOURCE_ID: 2025_ITEM_7_12]).
6. Pay close attention to the YEAR parameter for each piece of evidence to distinguish between historical filing sheets.
"""

def generate_answer(question, contexts, model="gpt-5.6"):
    client = OpenAI()

    evidence = "\n\n".join(
        f"SOURCE_ID: {c.get('chunk_id')}\n"
        f"YEAR: {c.get('year')}\n"
        f"SECTION: {c.get('section') or c.get('title')}\n"
        f"TEXT:\n{c.get('text')}"
        for c in contexts
    )

    prompt = f"""Question:
{question}

Evidence:
{evidence}

Answer the question using only the evidence above. Include source IDs for the claims you make."""

    
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    }

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content
