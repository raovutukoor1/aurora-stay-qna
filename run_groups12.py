import copy, json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / "notebooks" / "Aurora_Stays_RAG_POC_Local_executed.ipynb"
nb = nbformat.read(path.open("r", encoding="utf-8-sig"), as_version=4)
indices = [19, 24, 26, 29, 32, 35, 36, 40]
mini = nbformat.v4.new_notebook(metadata=copy.deepcopy(nb.metadata))
mini.cells = [copy.deepcopy(nb.cells[i]) for i in indices]
client = NotebookClient(mini, timeout=900, allow_errors=True, kernel_name="python3", resources={"metadata": {"path": str(root)}})
executed = client.execute()
for i, cell in zip(indices, executed.cells):
    nb.cells[i]["outputs"] = cell.get("outputs", [])
    nb.cells[i]["execution_count"] = cell.get("execution_count")
nb.metadata["last_execution_group"] = "Groups 1-2: imports, local paths, document conversion, and previews"
nbformat.write(nb, path.open("w", encoding="utf-8"))
print("Executed Groups 1-2")
for i in indices:
    print(i, nb.cells[i].get("execution_count"), len(nb.cells[i].get("outputs", [])))
