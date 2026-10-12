# Aurora Stays Q&A

Aurora Stays Q&A is a retrieval-augmented generation (RAG) application for answering questions about hotel policies. It retrieves relevant policy passages from a FAISS vector index and can return either an answer or an answer together with the supporting passages.

## Repository layout

```text
backend/
├── app.py
├── rag_utils.py
├── requirements.txt
├── Dockerfile
└── faiss_index/
    ├── index.faiss
    └── index.pkl
frontend/
├── app.py
├── requirements.txt
└── Dockerfile
```

## API endpoints

The Flask backend provides:

- `GET /` — health/welcome response
- `POST /v1/relevant_chunks` — returns the most relevant policy passages
- `POST /v1/answer_with_relevant_chunks` — returns an answer and supporting passages
- `POST /v1/answer` — returns only the generated answer

Example request:

```json
{
  "query": "What is the cancellation policy for a flexible reservation?",
  "k": 5,
  "temperature": 0.0,
  "max_tokens": 512
}
```

## Run locally

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The backend starts on `http://localhost:5000` by default. Set `OPENAI_API_KEY` and, if required, `OPENAI_BASE_URL` before using the answer endpoints.

### Frontend

The Streamlit frontend uses `http://localhost:5000` by default. Override this with the `BACKEND_URL` environment variable if the backend runs elsewhere.

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Docker

Each service includes its own `Dockerfile`. Build and run the backend and frontend from their respective directories. Configure the frontend backend URL and pass the required model/API environment variables when starting the containers.

## Notes

- The FAISS index is included under `backend/faiss_index`.
- Baseline retrieval setting: `k=5` chunks per query, selected from the dev/test retrieval comparison.
- Saved evaluation baseline: 60% dev pass rate and 30% held-out test pass rate with `k=5`.
- Do not commit API keys or GitHub tokens. Use environment variables or a local secrets file.
- The default embedding model can be overridden with the `EMBED_MODEL` environment variable.
