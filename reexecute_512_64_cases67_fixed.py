"""Run the held-out 512/64 evaluation with the cases 6/7 retrieval fixes."""

from copy import deepcopy
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "notebooks" / "Aurora_Stays_RAG_POC_Local_final_executed.ipynb"
OUTPUT = ROOT / "notebooks" / "Aurora_Stays_RAG_POC_Local_512_64_cases6_7_fixed_executed.ipynb"
REPORT = ROOT / "reports" / "notebook_512_64_cases6_7_fixed_cases.csv"


def find_cell(nb, number):
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.metadata.get("code_cell_number") == number:
            return cell
    raise KeyError(number)


def main():
    os.environ["CHUNK_SIZE"] = "512"
    os.environ["CHUNK_OVERLAP"] = "64"

    nb = nbformat.read(SOURCE.open("r", encoding="utf-8-sig"), as_version=4)

    # Avoid the Windows symlink requirement while reusing the local model cache.
    cell2 = find_cell(nb, 2)
    cell2.source = cell2.source.replace(
        'HF_CACHE = Path(os.environ.get("TEMP", str(PROJECT_ROOT))) / "aurora-hf-cache"',
        'HF_CACHE = Path(os.environ.get("USERPROFILE", str(PROJECT_ROOT))) / ".cache" / "huggingface"',
    )

    # Use the held-out test set, not the development dataframe.
    cell32 = find_cell(nb, 32)
    cell32.source = cell32.source.replace(
        "# Define the parameters for test case generation\n",
        "# Define the parameters for the held-out test case generation\n"
        'df = pd.read_excel("data/test.xlsx")\n'
        'print(f"Held-out evaluation rows: {len(df)}")\n\n',
    )

    # Make the copied notebook report the freshly executed result.
    cell36 = find_cell(nb, 36)
    cell36.source = '''# Code cell 036
import pandas as pd

thresholds = {
    "faithfulness": 0.85,
    "contextual recall": 0.80,
    "contextual precision": 0.75,
    "answer relevancy": 0.75,
}
rows = []
for index, (test_case, result) in enumerate(zip(test_cases, results.test_results), start=1):
    scores = {
        metric.name.replace(" [GEval]", "").strip().lower(): metric.score
        for metric in result.metrics_data
    }
    composite = (
        0.35 * scores["faithfulness"]
        + 0.25 * scores["contextual recall"]
        + 0.20 * scores["contextual precision"]
        + 0.20 * scores["answer relevancy"]
    )
    passed = all(scores[name] >= threshold for name, threshold in thresholds.items())
    rows.append({"case": index, "question": test_case.input, **scores, "composite": composite, "passed_all_metrics": passed})
evaluation_summary = pd.DataFrame(rows)
evaluation_summary.to_csv("reports/notebook_512_64_cases6_7_fixed_cases.csv", index=False)
print("Fresh 512/64 held-out evaluation with cases 6/7 retrieval fixes")
print(f"Pass rate: {int(evaluation_summary['passed_all_metrics'].sum())}/{len(evaluation_summary)}")
print(f"Composite: {evaluation_summary['composite'].mean():.3f}")
display(evaluation_summary.round(3))
'''
    cell37 = find_cell(nb, 37)
    cell37.source = '''# Code cell 037
display(evaluation_summary.round(3))
passed = int(evaluation_summary["passed_all_metrics"].sum())
print(f"Final held-out pass rate: {passed}/{len(evaluation_summary)} ({passed / len(evaluation_summary):.0%})")
print("Configuration: chunk_size=512, chunk_overlap=64, tokenizer/model max length=512")
print("Targeted fixes: property-specific breakfast routing and Advance Purchase evidence routing")
'''

    # Keep the copied notebook self-documenting: cell 40 is the deployment
    # source for backend/rag_utils.py, so update it with the tested backend.
    cell40 = find_cell(nb, 40)
    backend_source = (ROOT / "backend" / "rag_utils.py").read_text(encoding="utf-8")
    cell40.source = "%%writefile backend/rag_utils.py\n" + backend_source

    # Execute only through the local evaluation. Deployment/upload cells are
    # excluded and the original final notebook is never overwritten.
    last_index = max(
        i for i, cell in enumerate(nb.cells)
        if cell.cell_type == "code" and cell.metadata.get("code_cell_number") == 37
    )
    mini = nbformat.v4.new_notebook(metadata=deepcopy(nb.metadata))
    mini.cells = [deepcopy(cell) for cell in nb.cells[: last_index + 1]]
    client = NotebookClient(
        mini,
        timeout=1800,
        allow_errors=False,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()

    # Only after successful execution, merge the executed cells into a new
    # notebook copy and save it as the separate fixed-case artifact.
    for index, cell in enumerate(mini.cells):
        nb.cells[index] = deepcopy(cell)
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.metadata.get("code_cell_number", 0) >= 40:
            cell.outputs = []
            cell.execution_count = None
    nb.metadata["execution_scope"] = "512/64 held-out evaluation; deployment cells excluded"
    nb.metadata["targeted_fixes"] = [
        "property-specific breakfast candidate routing and context focus",
        "Advance Purchase candidate routing and context focus",
    ]
    nbformat.write(nb, OUTPUT)

    for cell in mini.cells:
        if cell.cell_type == "code" and cell.metadata.get("code_cell_number") == 36:
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream":
                    print("".join(output.get("text", [])), end="")
            break
    print(f"Saved executed notebook to {OUTPUT}")
    print(f"Saved case report to {REPORT}")


if __name__ == "__main__":
    main()
