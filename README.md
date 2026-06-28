# HireMesh — AI Recruitment Backend 🧠

HireMesh is an AI recruitment copilot that screens candidate resumes against a job description. Upload a CV and a job description, and a **team of AI agents** parses the resume, analyses the role, and returns a structured evaluation — a 0–100 fit score, key strengths, concerns, reasoning, and a hire recommendation — which a recruiter can then **interrogate with follow-up questions** in natural language.

This repository is the **backend service**: an asynchronous Python microservice (FastAPI + Celery/Redis + PostgreSQL) with a multi-agent orchestration layer (Agno) on top of Google Gemini and Groq. The product also has a Next.js frontend, which is **not** part of this repository — everything here is usable directly via the HTTP API.

---


## ✨ What it does

- **Resume screening.** Upload a PDF / DOCX / TXT resume plus a job description; get back a structured, explainable evaluation (score, strengths, concerns, reasoning, recommendation).
- **Multi-agent analysis.** A coordinator agent delegates to two specialists — a resume parser and a job analyst — and synthesises the final verdict.
- **Conversational follow-ups.** Ask free-text questions about any analysed candidate ("does this person have production Kubernetes experience?") and get a streamed, context-grounded answer.
- **Built to run as a service.** Async processing off the request path, per-IP rate limiting, crash/timeout handling, and full LLM observability.

---

## 🛠️ Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI · Uvicorn |
| Async tasks | Celery · Redis (broker) |
| Database | PostgreSQL · SQLAlchemy (psycopg 3) |
| AI orchestration | Agno (multi-agent teams) |
| LLMs | Google Gemini (analysis) · Groq `llama-3.3-70b` (follow-up) |
| Prompts & tracing | Langfuse (managed prompts + OpenTelemetry) |
| File parsing | pypdf · python-docx |
| Packaging | uv |
| Container / infra | Docker · docker-compose · Railway |

---

## ⚙️ How it works

```mermaid
flowchart LR
    Client[Client] -->|"POST /analyze"| API[FastAPI]
    Client -->|"GET /status (poll)"| API
    Client -->|"POST /followup (SSE)"| API

    API -->|enqueue task| Redis[(Redis broker)]
    API -->|read / write| DB[(PostgreSQL)]
    Redis -->|deliver task| Worker[Celery worker]
    Worker -->|read / write| DB

    Worker --> Team["Agno Team (Gemini)"]
    Team --> RP[Resume Parser]
    Team --> JA[Job Analyst]

    API --> Followup["Follow-up agent (Groq)"]
```

LLM analysis takes 10–30s, so it never runs on the request path. `POST /analyze` extracts the text, saves a row, **enqueues** a background task, and returns instantly with a `session_id`. A separate **Celery worker** runs the AI team and writes the result to PostgreSQL; the client **polls** `GET /status/{id}` until it's `completed`. Follow-up answers are **streamed** over Server-Sent Events. The API and worker never call each other directly — they're decoupled through Redis (task delivery) and PostgreSQL (shared state).

---

## 🚀 Getting started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (or pip)
- PostgreSQL and Redis — locally via the included `docker-compose.yml`, or hosted (e.g. Railway)
- API keys: **Google Gemini**, **Groq**, and **Langfuse** (public + secret)

### 1. Configure environment

Create a `.env` file in the project root:

```bash
# LLM providers
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key

# Infrastructure (Railway or local docker-compose)
DATABASE_URL=postgresql://user:pass@host:port/dbname
REDIS_URL=redis://default:pass@host:port

# Langfuse (managed prompts + tracing)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

> **Prompts are managed in Langfuse**, not hard-coded. The app fetches `resume-parser-instructions`, `job-analyst-instructions`, and `hr-team-lead-instructions` at startup and **fails fast** if they're missing — so those three prompts must exist in your Langfuse project.

### 2. Install dependencies

```bash
uv sync          # or: pip install -r requirements.txt
```

### 3. Run

The system needs **two processes** — the API and the worker:

```bash
# Terminal A — API server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal B — Celery worker (runs the AI analysis)
uv run celery -A app.tasks.celery_app worker --loglevel=info
```

The API is now at `http://localhost:8000` (interactive docs at `/docs`).

**Or run everything with Docker** — brings up PostgreSQL, Redis, the API, and the worker together:

```bash
docker compose up --build
```

---

## 📡 Usage examples

| Method | Endpoint | Purpose | Rate limit (per IP) |
|---|---|---|---|
| `POST` | `/analyze` | Upload a resume + job description; queue analysis | 5 / min |
| `GET` | `/status/{session_id}` | Poll status and fetch the result | 60 / min |
| `POST` | `/sessions/{session_id}/followup` | Ask a follow-up question (SSE stream) | 20 / min |

**1. Start an analysis** — returns a `session_id`:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "job_description=Senior Python backend engineer, FastAPI + async..." \
  -F "file=@resume.pdf"
# -> {"session_id": "a1b2c3…", "status": "processing"}
```

**2. Poll for the result:**

```bash
curl http://localhost:8000/status/a1b2c3…
# -> {
#      "status": "completed",
#      "result": {
#        "candidate_name": "Jane Doe",
#        "score": 92,
#        "key_strengths": ["...", "..."],
#        "concerns": ["..."],
#        "reasoning": "...",
#        "final_recommendation": "Strong Hire"
#      }
#    }
```

**3. Ask a follow-up** — streamed token-by-token over SSE:

```bash
curl -N -X POST http://localhost:8000/sessions/a1b2c3…/followup \
  -H "Content-Type: application/json" \
  -d '{"question": "Does this candidate have production Kubernetes experience?"}'
# -> data: {"delta": "Based"}
#    data: {"delta": " on the resume..."}
#    ...
#    data: {"done": true}
```

---

## 📂 Project structure

```
.
├── app/
│   ├── main.py          # FastAPI app: endpoints, file extraction, SSE follow-up
│   ├── tasks.py         # Celery worker: runs the AI team, parses & stores results
│   ├── agents.py        # Agno agents/team, model config, Langfuse prompts + tracing
│   ├── database.py      # SQLAlchemy models, engine, self-healing migration
│   ├── schemas.py       # Pydantic models (CandidateEvaluation, FollowupRequest)
│   └── rate_limit.py    # Redis-backed per-IP rate limiter
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```
