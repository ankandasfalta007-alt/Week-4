# NexisCode: GenAI coding assistant

## Architecture
Browser (index.html) -> FastAPI -> [JWT auth] -> [input guardrails] -> [TF-IDF RAG over your uploads]
-> [last 10 turns of memory] -> Claude (streaming, optional image) -> [output redaction] -> SQLite (messages, usage)

| Requirement | Where |
|---|---|
| RAG | `/ingest` (PDF/text, chunked 900/150) + `retrieve()` |
| Memory | `messages` table, last 10 turns per `conv` |
| Multimodal | `/chat` accepts an image; `/ingest` accepts PDFs |
| Guardrails | `check_input()` (injection, malware, length), `redact()` (keys), system prompt |
| Streaming | `StreamingResponse` over `messages.stream` |
| Auth | PBKDF2 passwords + JWT (24h) |
| Analytics | `/analytics` + sidebar (tokens, latency, blocked) |

## Run locally
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=... JWT_SECRET=$(openssl rand -hex 32)
    uvicorn main:app --reload      # http://localhost:8000

## Deploy (Railway)
1. Push this folder to GitHub. 2. Railway -> New Project -> Deploy from repo (Dockerfile is detected).
3. Set `ANTHROPIC_API_KEY`, `JWT_SECRET`, and `DB_PATH=/data/nexis.db`; attach a volume at `/data`.
4. Generate a public domain. Health check path: `/health`.
(Vercel suits the frontend only; this backend needs a long-lived streaming server. AWS: same image on App Runner/ECS.)

## Before real users
- Swap TF-IDF for embeddings + pgvector; move SQLite to Postgres.
- Add rate limiting, HTTPS-only CORS config, refresh tokens, and a moderation classifier on top of the regex rules.
- Add evals for retrieval quality and guardrail false positives.
