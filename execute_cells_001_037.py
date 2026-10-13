import copy
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "Aurora_Stays_RAG_POC_Local_executed.ipynb"


def source_text(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def output_summary(outputs):
    parts = []
    for output in outputs:
        kind = output.get("output_type")
        if kind == "error":
            parts.append(f"ERROR {output.get('ename')}: {output.get('evalue')}")
        elif kind in {"stream", "execute_result", "display_data"}:
            data = output.get("text") or output.get("data", {}).get("text/plain") or output.get("data", {}).get("text/markdown")
            if isinstance(data, list):
                data = "".join(data)
            if data:
                line = " ".join(str(data).split())
                parts.append(line[:220])
    if not parts:
        return "No displayed output; cell completed successfully."
    summary = " | ".join(parts)
    return summary[:500]


notebook = nbformat.read(NOTEBOOK_PATH.open("r", encoding="utf-8-sig"), as_version=4)
code_cells = {
    int(cell.metadata["code_cell_number"]): (index, cell)
    for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "code" and "code_cell_number" in cell.metadata
}

# Make the optional preview deterministic so every code cell through 037 can run.
preview_index, preview_cell = code_cells[7]
preview_cell.source = """# Code cell 007
# Purpose: Select a policy document for the Markdown preview.
path = str(Path(root_path) / "aurorastay_room_and_property_information.pdf")
print(f"Preview document: {path}")
"""

selected_numbers = list(range(1, 38))
mini = nbformat.v4.new_notebook(metadata=copy.deepcopy(notebook.metadata))
mini.cells = [copy.deepcopy(code_cells[number][1]) for number in selected_numbers]

# Clear prior outputs and execution counts for this fresh sequential run.
for number in selected_numbers:
    cell = code_cells[number][1]
    cell.outputs = []
    cell.execution_count = None

client = NotebookClient(
    mini,
    timeout=1800,
    allow_errors=False,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
)

with client.setup_kernel():
    for position, number in enumerate(selected_numbers):
        print(f"Executing code cell {number:03d}...", flush=True)
        executed = client.execute_cell(mini.cells[position], position)
        original_index, original = code_cells[number]
        executed.execution_count = number
        summary = output_summary(executed.get("outputs", []))
        executed.metadata["execution_summary"] = summary
        executed.outputs.append(
            nbformat.v4.new_output(
                output_type="display_data",
                data={
                    "text/markdown": [f"**Output summary:** {summary}"],
                    "text/plain": [f"Output summary: {summary}"],
                },
                metadata={"generated_by": "execute_cells_001_037.py"},
            )
        )
        notebook.cells[original_index] = copy.deepcopy(executed)
        notebook.metadata["last_execution_group"] = f"Sequential execution through code cell {number:03d}"
        nbformat.write(notebook, NOTEBOOK_PATH.open("w", encoding="utf-8"))
        safe_summary = summary.encode("ascii", "replace").decode("ascii")
        print(f"Completed code cell {number:03d}: {safe_summary}", flush=True)

print("Completed sequential execution through code cell 037.")
