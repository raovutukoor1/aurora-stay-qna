import copy
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / "notebooks" / "Aurora_Stays_RAG_POC_Local_executed.ipynb"
nb = nbformat.read(path.open("r", encoding="utf-8-sig"), as_version=4)
indices = [19, 24, 26, 29, 32, 35, 36, 46, 50, 55, 57, 67, 71, 75, 79]
mini = nbformat.v4.new_notebook(metadata=copy.deepcopy(nb.metadata))
mini.cells = [copy.deepcopy(nb.cells[i]) for i in indices]
client = NotebookClient(mini, timeout=1800, allow_errors=True, kernel_name="python3", resources={"metadata": {"path": str(root)}})
executed = client.execute()
for i, cell in zip(indices, executed.cells):
    nb.cells[i]["outputs"] = cell.get("outputs", [])
    nb.cells[i]["execution_count"] = cell.get("execution_count")
nb.metadata["last_execution_group"] = "Group 3 rerun with sentence-transformers 6.1.0"
nbformat.write(nb, path.open("w", encoding="utf-8"))
print("Reran Group 3 embedding and retrieval")
for i in indices:
    print(i, nb.cells[i].get("execution_count"), len(nb.cells[i].get("outputs", [])))
