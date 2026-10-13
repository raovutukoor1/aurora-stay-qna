import re
from pathlib import Path

import truststore
from github import Github


truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parent
token_text = (ROOT / "Supplied files" / "Access Token Git Hub.txt").read_text(encoding="utf-8-sig")
matches = re.findall(r"(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)", token_text)
token = matches[-1] if matches else [line.strip() for line in token_text.splitlines() if line.strip()][-1].split()[-1]

repo_name = "aurora-stay-qna-codespaces"
files = {
    "README.md": None,
    "notebooks/Aurora_Stays_RAG_POC_GitHub_Codespaces.ipynb": ROOT / "notebooks" / "Aurora_Stays_RAG_POC_GitHub_Codespaces.ipynb",
    "backend/app.py": ROOT / "backend" / "app.py",
    "backend/rag_utils.py": ROOT / "backend" / "rag_utils.py",
    "backend/requirements.txt": ROOT / "backend" / "requirements.txt",
    "backend/Dockerfile": ROOT / "backend" / "Dockerfile",
    "backend/faiss_index/index.faiss": ROOT / "backend" / "faiss_index" / "index.faiss",
    "backend/faiss_index/index.pkl": ROOT / "backend" / "faiss_index" / "index.pkl",
    "frontend/app.py": ROOT / "frontend" / "app.py",
    "frontend/requirements.txt": ROOT / "frontend" / "requirements.txt",
    "frontend/Dockerfile": ROOT / "frontend" / "Dockerfile",
    "data/evaluation.csv": ROOT / "data" / "evaluation.csv",
    "data/test.xlsx": ROOT / "data" / "test.xlsx",
    "data/aurorastay_booking_terms_and_cancellation_policy.pdf": ROOT / "data" / "aurorastay_booking_terms_and_cancellation_policy.pdf",
    "data/aurorastay_guest_conduct_accessibility_and_resolution_guidelines.pdf": ROOT / "data" / "aurorastay_guest_conduct_accessibility_and_resolution_guidelines.pdf",
    "data/aurorastay_refunds_and_additional_charges_guide.pdf": ROOT / "data" / "aurorastay_refunds_and_additional_charges_guide.pdf",
    "data/aurorastay_room_and_property_information.pdf": ROOT / "data" / "aurorastay_room_and_property_information.pdf",
}

readme = """# Aurora Stays Q&A — GitHub Codespaces

This repository contains the K=5 Aurora Stays RAG baseline, including the Codespaces-ready notebook, policy documents, evaluation data, FAISS index, Flask backend, and Streamlit frontend.

## Codespaces quick start

1. Create a Codespace from the `main` branch.
2. Add `OPENAI_API_KEY` and, if needed, `OPENAI_BASE_URL` as Codespaces secrets.
3. Run the backend:

```bash
cd backend
docker build -t aurora-stays-backend .
docker run --rm -p 7860:7860 -e OPENAI_API_KEY -e OPENAI_BASE_URL aurora-stays-backend
```

4. Set port 7860 to Public and use its forwarded URL as `BACKEND_URL`.
5. Open `notebooks/Aurora_Stays_RAG_POC_GitHub_Codespaces.ipynb` and run the notebook from the repository root.

The notebook uses environment variables for credentials, `test.xlsx` for the held-out test set, and `k=5` consistently. Credentials and local configuration files are intentionally excluded.

The backend exposes `/v1/relevant_chunks`, `/v1/answer_with_relevant_chunks`, and `/v1/answer`. The frontend Dockerfile listens on port 8501.
"""

github = Github(login_or_token=token)
user = github.get_user()
try:
    repo = user.get_repo(repo_name)
    print(f"Using existing repository: {repo.html_url}")
except Exception as exc:
    if "404" not in str(exc):
        raise
    repo = user.create_repo(
        repo_name,
        description="Aurora Stays RAG baseline configured for GitHub Codespaces",
        private=False,
        auto_init=True,
    )
    print(f"Created repository: {repo.html_url}")

for remote_path, local_path in files.items():
    content = readme.encode("utf-8") if local_path is None else local_path.read_bytes()
    try:
        existing = repo.get_contents(remote_path, ref="main")
        repo.update_file(
            path=remote_path,
            message="Refresh Codespaces-ready K=5 baseline",
            content=content,
            sha=existing.sha,
            branch="main",
        )
        print(f"Updated {remote_path}")
    except Exception as exc:
        if "404" not in str(exc):
            raise
        repo.create_file(
            path=remote_path,
            message="Add Codespaces-ready K=5 baseline",
            content=content,
            branch="main",
        )
        print(f"Created {remote_path}")

print(f"Repository URL: {repo.html_url}")
