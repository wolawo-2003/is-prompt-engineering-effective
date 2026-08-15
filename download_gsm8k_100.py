import json
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
N = 100
OUTPUT = Path(__file__).parent / "gsm8k_test_100.jsonl"

print("Downloading GSM8K test set (1319 items), keeping first 100 ...")
with urllib.request.urlopen(URL, timeout=60) as resp:
    data = resp.read().decode("utf-8")

lines = data.splitlines()
print(f"Downloaded. Test set has {len(lines)} items.")

items = [json.loads(line) for line in lines[:N]]
with OUTPUT.open("w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved first {len(items)} items to: {OUTPUT}")

print("\nPreview (first 3):")
for i, item in enumerate(items[:3], 1):
    print(f"[{i}] Q: {item['question'][:70]}...")
    print(f"     A(final): {item['answer'].split('####')[-1].strip()}")