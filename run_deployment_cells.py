import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "Aurora_Stays_RAG_POC_Local_executed.ipynb"


def patch_source(number: int, source: str) -> str:
    replacements = {
        48: '''# Code cell 048
# Load the GitHub token from the supplied local file without displaying it.
token_path = PROJECT_ROOT / "Supplied files" / "Access Token Git Hub.txt"
GITHUB_TOKEN = config.get("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN and token_path.exists():
    GITHUB_TOKEN = token_path.read_text(encoding="utf-8").strip()
if not GITHUB_TOKEN:
    raise ValueError("No GitHub token is available.")
''',
        50: '''# Code cell 050
# Use the local backend for endpoint verification because no Codespaces URL was supplied.
model_root_url = "http://127.0.0.1:5000"
''',
        53: '''# Code cell 053
payload = {
    "query": "What is the cancellation policy for a flexible reservation?",
    "k": 3,
}
''',
        57: '''# Code cell 057
payload = {
    "query": "What is the cancellation policy for a flexible reservation?",
    "k": 3,
    "model_name": "gpt-4o-mini",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
}
''',
    }
    return replacements.get(number, source)


def main() -> None:
    supplied_config = ROOT / "Supplied files" / "confignew.txt"
    if supplied_config.exists():
        for key, value in json.loads(supplied_config.read_text(encoding="utf-8-sig")).items():
            if value:
                os.environ[key] = str(value)
    notebook = nbformat.read(NOTEBOOK_PATH.open("r", encoding="utf-8-sig"), as_version=4)
    by_number = {
        int(cell.metadata["code_cell_number"]): (index, cell)
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and "code_cell_number" in cell.metadata
    }
    for number, (_, cell) in by_number.items():
        cell.source = patch_source(number, cell.source)

    bootstrap = nbformat.v4.new_code_cell(
        """import json, os, requests
from pathlib import Path
PROJECT_ROOT = Path.cwd()
CONFIG_PATH = PROJECT_ROOT / 'config' / 'config.json'
config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
supplied = PROJECT_ROOT / 'Supplied files' / 'confignew.txt'
if not config.get('OPENAI_API_KEY') and supplied.exists():
    config.update(json.loads(supplied.read_text(encoding='utf-8-sig')))
os.environ['OPENAI_API_KEY'] = config.get('OPENAI_API_KEY', '')
os.environ['OPENAI_BASE_URL'] = config.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
"""
    )
    # Run the GitHub upload cell and the remaining deployment tests locally.
    numbers = [48, 49] + list(range(50, 60 + 1))
    mini = nbformat.v4.new_notebook(metadata=copy.deepcopy(notebook.metadata))
    mini.cells = [bootstrap] + [copy.deepcopy(by_number[number][1]) for number in numbers]
    client = NotebookClient(
        mini,
        timeout=600,
        allow_errors=False,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    server = None
    with client.setup_kernel():
        client.execute_cell(mini.cells[0], 0)
        for position, number in enumerate(numbers, start=1):
            if number == 54:
                server_env = os.environ.copy()
                server_env["PYTHONPATH"] = str(ROOT / "backend")
                server_env["EMBED_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
                server_env["HF_HUB_OFFLINE"] = "1"
                server_env["TRANSFORMERS_OFFLINE"] = "1"
                server = subprocess.Popen(
                    [sys.executable, "app.py"],
                    cwd=str(ROOT / "backend"),
                    env=server_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(30)
            print(f"Executing code cell {number:03d}...", flush=True)
            executed = client.execute_cell(mini.cells[position], position)
            original_index, _ = by_number[number]
            notebook.cells[original_index] = copy.deepcopy(executed)
            notebook.metadata["last_execution_group"] = "Deployment and local endpoint checks through code cell 060"
            nbformat.write(notebook, NOTEBOOK_PATH.open("w", encoding="utf-8"))
            print(f"Completed code cell {number:03d}; outputs={len(executed.get('outputs', []))}", flush=True)
    if server is not None:
        server.terminate()
        server.wait(timeout=10)
    print("Completed deployment/local endpoint cells 048-060.", flush=True)


if __name__ == "__main__":
    main()
