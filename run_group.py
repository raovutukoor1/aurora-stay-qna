import copy
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
source_path = root / "notebooks" / "Aurora_Stays_RAG_POC_Local.ipynb"
output_path = root / "notebooks" / "Aurora_Stays_RAG_POC_Local_executed.ipynb"

source = nbformat.read(source_path.open("r", encoding="utf-8-sig"), as_version=4)
selected = [17, 19, 24, 26]
mini = nbformat.v4.new_notebook(metadata=copy.deepcopy(source.metadata))
mini.cells = [copy.deepcopy(source.cells[i]) for i in selected]

client = NotebookClient(
    mini,
    timeout=900,
    allow_errors=True,
    kernel_name="python3",
    resources={"metadata": {"path": str(root)}},
)
executed = client.execute()

result = copy.deepcopy(source)
if output_path.exists():
    existing = nbformat.read(output_path.open("r", encoding="utf-8-sig"), as_version=4)
else:
    existing = result

for source_index, executed_cell in zip(selected, executed.cells):
    result.cells[source_index].outputs = executed_cell.get("outputs", [])
    result.cells[source_index].execution_count = executed_cell.get("execution_count")

result.metadata["execution_note"] = (
    "Group 1 executed locally: dependency setup, imports, project-root resolution, and PDF path discovery."
)
nbformat.write(result, output_path.open("w", encoding="utf-8"))
print("Executed Group 1: cells 001-004")
for index in selected:
    cell = result.cells[index]
    print(index, "execution_count=", cell.execution_count, "outputs=", len(cell.outputs))
