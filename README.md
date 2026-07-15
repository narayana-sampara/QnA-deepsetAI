# QnA API

A FastAPI service that answers free-text questions by combining a local
extractive question-answering model (`deepset/roberta-base-squad2`) with
live web search results. It routes each question through specialized
handlers for arithmetic, yes/no claims, and general factual questions, then
uses a fuzzy-matching consensus engine across multiple web snippets to pick
the most reliable answer.

## How it works

1. **Math questions** (e.g. `12 * 4`) are parsed and evaluated directly with
   `ast`, no web lookup needed.
2. **Yes/No claims** (questions starting with "is", "are", "was", "were",
   "do", "does") are answered by running web QA and checking whether the
   consensus context contains a negation.
3. **General questions** are searched via DuckDuckGo (`ddgs`), each result
   snippet is normalized and fed to the QA model, and the resulting
   candidate answers are clustered with fuzzy string matching
   (`rapidfuzz`) to find a consensus answer, a confidence label, and
   supporting sources.
4. Role-based questions (Governor / Chief Minister / Prime Minister) get an
   auto-generated explanatory summary instead of a raw snippet.

## Project structure

```
app/
  main.py        FastAPI app: middleware, auth, rate limiting, /ask endpoint
  qna_core.py     Core QA/search/consensus logic and the loaded model pipeline
  config.py       Environment-driven settings
QnA_ai.py         Standalone CLI version of the same QA engine (for local testing)
Dockerfile        Multi-stage build, runs as non-root user under gunicorn
docker-compose.yml
requirements.txt
.env.example      Template for required environment variables
```

## API

### `POST /ask`
Requires the `X-API-Key` header. Rate-limited (default `20/minute`).

Request body:
```json
{ "question": "Who is the Prime Minister of India?" }
```

Response:
```json
{
  "answer": "Narendra Modi",
  "confidence": "High",
  "consensus": "4/9",
  "status": "Verified",
  "summary": "...",
  "sources": ["https://..."]
}
```

### `GET /health`
Liveness check, returns `{"status": "ok"}`.

### `GET /ready`
Readiness check, returns whether the QA model has finished loading.

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Description | Default |
|---|---|---|
| `QNA_API_KEY` | Required API key clients must send in `X-API-Key` | *(none — service refuses requests if unset)* |
| `QNA_ALLOWED_ORIGINS` | Comma-separated CORS origins | *(empty)* |
| `QNA_ALLOWED_HOSTS` | Comma-separated allowed `Host` headers | `*` |
| `QNA_RATE_LIMIT` | slowapi rate limit expression | `20/minute` |
| `QNA_MAX_QUESTION_LENGTH` | Max characters accepted in `question` | `300` |
| `QNA_LOG_LEVEL` | Logging level | `INFO` |

Copy `.env.example` to `.env` and fill in real values before running.

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env   # edit values
uvicorn app.main:app --host 0.0.0.0 --port 8891
```

Or try the standalone CLI version (no API/auth, interactive prompt):

```bash
python QnA_ai.py
```

## Running with Docker

```bash
docker compose up --build
```

The service listens on port `8891` and exposes `/health` and `/ready` for
container health checks. The container runs as a non-root user with a
read-only root filesystem and no extra Linux capabilities.

## Notes

- The QA model runs on GPU automatically if CUDA is available, otherwise CPU.
- API docs (`/docs`, `/redoc`, `/openapi.json`) are disabled in production.
