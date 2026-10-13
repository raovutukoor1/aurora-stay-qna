import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parent / "notebooks" / "Aurora_Stays_RAG_POC_Local_executed.ipynb"

NOISE_MARKERS = (
    "Failed to multipart ingest runs",
    "Failed to send compressed multipart ingest",
    "Task was destroyed",
    "Exception in callback Task.__step__",
    "RuntimeWarning:",
    "DeprecationWarning:",
    "FutureWarning:",
    "Proactor event loop does not implement add_reader",
)


def output_text(output):
    if output.get("output_type") == "stream":
        value = output.get("text", "")
        return "".join(value) if isinstance(value, list) else str(value)
    data = output.get("data", {})
    parts = []
    for key in ("text/plain", "text/html"):
        value = data.get(key, "")
        parts.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(parts)


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8-sig"))
removed = 0
for cell in notebook.get("cells", []):
    kept = []
    for output in cell.get("outputs", []):
        text = output_text(output)
        is_widget = "application/vnd.jupyter.widget-view+json" in output.get("data", {})
        is_blank = not text.strip() and not is_widget
        is_noise = any(marker in text for marker in NOISE_MARKERS)
        if is_widget or is_blank or is_noise:
            removed += 1
        else:
            kept.append(output)
    cell["outputs"] = kept

NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Removed {removed} irrelevant notebook outputs")
