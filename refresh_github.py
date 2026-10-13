import re
from pathlib import Path

import truststore
from github import Github

truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parent
token_text = (ROOT / "Supplied files" / "Access Token Git Hub.txt").read_text(encoding="utf-8-sig")
matches = re.findall(r"(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)", token_text)
if matches:
    token = matches[-1]
else:
    lines = [line.strip() for line in token_text.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("GitHub access-token file is empty")
    token = lines[-1].split()[-1]

repo_name = "aurora-stay-qna"
files = {
    "backend/app.py": ROOT / "backend" / "app.py",
    "backend/rag_utils.py": ROOT / "backend" / "rag_utils.py",
    "backend/requirements.txt": ROOT / "backend" / "requirements.txt",
    "backend/Dockerfile": ROOT / "backend" / "Dockerfile",
    "frontend/app.py": ROOT / "frontend" / "app.py",
    "frontend/requirements.txt": ROOT / "frontend" / "requirements.txt",
    "frontend/Dockerfile": ROOT / "frontend" / "Dockerfile",
    "README.md": ROOT / "README.md",
    # Replace the repository's prior notebook with the Codespaces-ready version
    # while keeping the established remote path used by the assignment.
    "notebooks/Aurora_Stays_RAG_POC_Local_executed.ipynb": ROOT / "notebooks" / "Aurora_Stays_RAG_POC_GitHub_Codespaces.ipynb",
}

github = Github(login_or_token=token)
user = github.get_user()
repo = user.get_repo(repo_name)
print(f"Authenticated as {user.login}; refreshing {repo.full_name}")

for remote_path, local_path in files.items():
    content = local_path.read_bytes()
    try:
        existing = repo.get_contents(remote_path, ref="main")
        repo.update_file(
            path=remote_path,
            message="Refresh K=5 baseline",
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
            message="Add K=5 baseline file",
            content=content,
            branch="main",
        )
        print(f"Created {remote_path}")

print(f"Repository: {repo.html_url}")
