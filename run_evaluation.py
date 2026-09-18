import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.rag import SECRAG
from src.evaluate import run_evaluation


load_dotenv()

with open("evaluation/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

rag = SECRAG()
rows = run_evaluation(rag, questions)

df = pd.DataFrame(rows)
Path("evaluation").mkdir(exist_ok=True)
df.to_csv("evaluation/results.csv", index=False)

print("\nEvaluation results:")
print(df.to_string(index=False))

numeric = df.select_dtypes(include="number")
if not numeric.empty:
    print("\nMean metrics:")
    print(numeric.mean(numeric_only=True).to_string())
