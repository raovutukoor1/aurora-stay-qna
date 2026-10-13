import json
from pathlib import Path

path = Path(__file__).resolve().parent / "notebooks" / "Aurora_Stays_RAG_POC_Local_executed.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8-sig"))
changed = 0
cleared = 0

for cell in notebook["cells"]:
    source = "".join(cell.get("source", []))
    if '"k": 3' in source:
        cell["source"] = source.replace('"k": 3', '"k": 5').splitlines(keepends=True)
        changed += 1
    if "Code cell 053" in source or "Code cell 057" in source:
        if cell.get("outputs"):
            cleared += len(cell["outputs"])
            cell["outputs"] = []
        cell["execution_count"] = None
    output_text = json.dumps(cell.get("outputs", []), ensure_ascii=False)
    if "'k': 3" in output_text or '"k": 3' in output_text:
        cleared += len(cell.get("outputs", []))
        cell["outputs"] = []
        cell["execution_count"] = None

path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Changed {changed} payload cells to k=5; cleared {cleared} stale outputs")
