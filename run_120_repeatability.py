"""Run the held-out evaluation repeatedly and record consistency findings."""

from copy import deepcopy
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "120_token_option" / "Aurora_Stays_RAG_POC_Local_120_16_token_safe_executed.ipynb"
OUTPUT = ROOT / "120_token_option" / "Aurora_Stays_RAG_POC_Local_120_16_token_safe_repeatability.ipynb"
REPORT = ROOT / "reports" / "notebook_120_16_repeatability_findings.md"
REPEATS = 3

LOAD_EXISTING_INDEX = '''# Code cell 017
# Load the existing 120/16 FAISS index; index construction is intentionally skipped.
vectorstore = FAISS.load_local(
    "backend/faiss_index_120_16_token_safe",
    embeddings,
    allow_dangerous_deserialization=True,
)
print(f"Loaded existing FAISS vectors: {vectorstore.index.ntotal}")
'''


def cell_positions(notebook):
    return {
        int(cell.metadata["code_cell_number"]): index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and "code_cell_number" in cell.metadata
    }


def execute_cell(client, notebook, positions, number):
    position = positions[number]
    print(f"Executing code cell {number:03d}...", flush=True)
    executed = client.execute_cell(notebook.cells[position], position)
    executed.execution_count = number
    notebook.cells[position] = executed


def main():
    notebook = nbformat.read(SOURCE.open("r", encoding="utf-8"), as_version=4)
    repeated = deepcopy(notebook)
    positions = cell_positions(repeated)

    for cell in repeated.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    repeated.cells[positions[17]].source = LOAD_EXISTING_INDEX
    client = NotebookClient(
        repeated,
        timeout=1800,
        allow_errors=False,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )

    with client.setup_kernel():
        # Establish the same environment and evaluation definitions once.
        for number in list(range(1, 17)) + list(range(17, 32)):
            execute_cell(client, repeated, positions, number)

        for repeat in range(1, REPEATS + 1):
            print(f"===== evaluation repeat {repeat}/{REPEATS} =====", flush=True)
            for number in (32, 33, 34, 35, 36):
                execute_cell(client, repeated, positions, number)

            # Persist the complete case-level result before the next repeat.
            capture = nbformat.v4.new_code_cell(
                source=(
                    "evaluation_summary.to_csv("
                    f"'reports/notebook_120_16_repeat_{repeat}.csv', index=False)"
                )
            )
            client.execute_cell(capture, -1)

        execute_cell(client, repeated, positions, 37)

    frames = []
    for repeat in range(1, REPEATS + 1):
        path = ROOT / "reports" / f"notebook_120_16_repeat_{repeat}.csv"
        frame = pd.read_csv(path)
        frame["repeat"] = repeat
        frames.append(frame)
    all_results = pd.concat(frames, ignore_index=True)

    metric_columns = [
        "faithfulness",
        "contextual recall",
        "contextual precision",
        "answer relevancy",
        "composite",
    ]
    pass_rates = [float(frame["passed_all_metrics"].mean()) for frame in frames]
    avg_composites = [float(frame["composite"].mean()) for frame in frames]
    pivot = all_results.pivot(index="case", columns="repeat", values=metric_columns)
    max_metric_delta = 0.0
    metric_delta_lines = []
    for metric in metric_columns:
        values = all_results.groupby("case")[metric].agg(["min", "max"])
        delta = float((values["max"] - values["min"]).max())
        max_metric_delta = max(max_metric_delta, delta)
        metric_delta_lines.append(f"- {metric}: maximum case-level range {delta:.4f}")

    pass_sets = [
        set(frame.loc[frame["passed_all_metrics"].astype(bool), "case"].astype(int))
        for frame in frames
    ]
    stable_pass_cases = sorted(set.intersection(*pass_sets))
    stable_fail_cases = sorted(set.intersection(*[set(frame["case"].astype(int)) - passes for frame, passes in zip(frames, pass_sets)]))
    varying_cases = sorted(set.union(*pass_sets) - set.intersection(*pass_sets))
    pass_rate_consistent = len(set(round(value, 10) for value in pass_rates)) == 1

    summary_lines = [
        "# 120/16 held-out evaluation repeatability findings",
        "",
        f"Evaluation repeated {REPEATS} times with the same configuration: K=5, temperature=0, top_p=1, max_tokens=512, 120-token chunks, 16-token overlap, and the existing FAISS index.",
        "",
        "## Run comparison",
        "",
        "| Run | Passed | Failed | Pass rate | Average composite |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index, (frame, pass_rate, avg_composite) in enumerate(zip(frames, pass_rates, avg_composites), 1):
        passed = int(frame["passed_all_metrics"].sum())
        summary_lines.append(f"| {index} | {passed} | {len(frame) - passed} | {pass_rate:.1%} | {avg_composite:.4f} |")

    summary_lines += [
        "",
        f"Pass-rate consistency: **{'consistent' if pass_rate_consistent else 'not consistent'}**.",
        f"Stable passing cases across all runs: {', '.join(map(str, stable_pass_cases)) or 'none'}.",
        f"Stable failing cases across all runs: {', '.join(map(str, stable_fail_cases)) or 'none'}.",
        f"Cases whose pass/fail status varied: {', '.join(map(str, varying_cases)) or 'none'}.",
        "",
        "## Maximum case-level metric ranges across runs",
        "",
        *metric_delta_lines,
        "",
        "## Interpretation",
        "",
        "The three runs use deterministic generation settings, but the evaluation uses model-judged metrics. Therefore, small score movement is expected even when the retrieved context and generated answer are unchanged. The pass/fail conclusion is considered operationally consistent when the pass rate and case-level pass sets remain unchanged.",
        "",
        "The detailed case-level results are saved as `reports/notebook_120_16_repeat_1.csv`, `reports/notebook_120_16_repeat_2.csv`, and `reports/notebook_120_16_repeat_3.csv`.",
    ]
    report_text = "\n".join(summary_lines) + "\n"
    REPORT.write_text(report_text, encoding="utf-8")

    repeated.metadata["repeatability_runs"] = REPEATS
    repeated.metadata["repeatability_report"] = str(REPORT.relative_to(ROOT))
    repeated.metadata["repeatability_pass_rates"] = [round(value, 6) for value in pass_rates]
    repeated.metadata["repeatability_max_metric_delta"] = round(max_metric_delta, 6)
    repeated.cells.append(nbformat.v4.new_markdown_cell(report_text))
    nbformat.write(repeated, OUTPUT.open("w", encoding="utf-8"))

    print(f"Wrote repeatability notebook: {OUTPUT}")
    print(f"Wrote findings report: {REPORT}")
    print(f"Pass rates: {[round(value, 4) for value in pass_rates]}")
    print(f"Stable pass cases: {stable_pass_cases}")
    print(f"Stable fail cases: {stable_fail_cases}")
    print(f"Varying cases: {varying_cases}")
    print(f"Maximum case-level metric delta: {max_metric_delta:.4f}")


if __name__ == "__main__":
    main()
