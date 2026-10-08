"""Append one appraisal to an answers file: python scripts/answer.py answers.json KEY '{"value": ...}'"""
import json
import sys
from pathlib import Path

path, key, data = Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
answers = json.loads(path.read_text()) if path.exists() else {}
answers[key] = data
path.write_text(json.dumps(answers, indent=1, ensure_ascii=False))
print(f"saved {key} ({len(answers)} answers)")
