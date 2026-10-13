import json
from pathlib import Path

root = Path(__file__).resolve().parent
paths = [
    root / "notebooks" / "Aurora_Stays_RAG_POC_Local.ipynb",
    root / "notebooks" / "Aurora_Stays_RAG_POC_Local_executed.ipynb",
]

for path in paths:
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    cells = []
    number = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "markdown" and cell.get("metadata", {}).get("code_cell_number_marker"):
            continue
        if cell.get("cell_type") == "code":
            number += 1
            marker = {
                "cell_type": "markdown",
                "metadata": {"code_cell_number_marker": True},
                "source": [f"### Code cell {number:03d}\n"],
            }
            cells.append(marker)
        cells.append(cell)
    notebook["cells"] = cells
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path.name, "visible markers:", number)
