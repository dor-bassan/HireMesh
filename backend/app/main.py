"""FastAPI application — the synchronous, request-facing side of the system.

Three endpoints:
  * ``POST /analyze``                        -> extract resume text, save a row,
                                                queue the background analysis,
                                                return a ``session_id`` immediately.
  * ``GET  /status/{session_id}``            -> poll the analysis status/result.
  * ``POST /sessions/{session_id}/followup`` -> stream a context-grounded answer (SSE).

This process stays fast: the slow AI analysis runs in the Celery worker (see
``tasks.py``). The two communicate only through Redis (task queue) and PostgreSQL.
"""
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import uuid
import json
import logging
import uvicorn
import io
from datetime import datetime, timedelta
import pypdf
import docx

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse
from app.database import init_db, SessionLocal, AnalysisResult
from app.tasks import process_resume_analysis
from app.agents import followup_agent
from app.schemas import FollowupRequest
from app.rate_limit import rate_limiter
from agno.run.agent import RunEvent

load_dotenv()

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="HireMesh API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # For local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


# --- Extraction functions inside the API ---
async def extract_text_from_upload(file: UploadFile) -> str:
    """Extract plain text from an uploaded PDF / DOCX / TXT, in memory.

    The file is never written to disk. Returns ``""`` on any extraction error or
    for unsupported types, leaving the caller to reject empty results.
    """
    # Read the file into memory (Bytes)
    content = await file.read()
    file_obj = io.BytesIO(content)
    text = ""

    try:
        filename = file.filename.lower()
        if filename.endswith('.pdf'):
            reader = pypdf.PdfReader(file_obj)
            for page in reader.pages:
                text += page.extract_text() + "\n"

        elif filename.endswith('.docx'):
            doc = docx.Document(file_obj)
            for para in doc.paragraphs:
                text += para.text + "\n"

        elif filename.endswith('.txt'):
            text = content.decode('utf-8')

    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""

    return text


# ---------------------------------

@app.post("/analyze", dependencies=[Depends(rate_limiter(limit=5, window_seconds=60))])
async def start_analysis(job_description: str = Form(...), file: UploadFile = File(...)):
    """Start a screening job.

    Extracts the resume text, persists a ``pending`` row, enqueues the analysis
    on the worker, and returns a ``session_id`` right away — the AI work happens
    asynchronously. The client then polls ``GET /status/{session_id}``.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"🔵 NEW REQUEST: {session_id}")

    # 1. Extract the text in memory (without saving a file)
    resume_text = await extract_text_from_upload(file)

    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file")

    # 2. Save to DB
    db = SessionLocal()
    try:
        new_analysis = AnalysisResult(
            session_id=session_id,
            status="pending",
            job_description=job_description,
            resume_text=resume_text,
        )
        db.add(new_analysis)
        db.commit()
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return {"error": "Database error"}
    finally:
        db.close()

    # 3. Send the text (not the file) to Worker
    try:
        # Sending the text as an argument
        process_resume_analysis.delay(session_id, resume_text, job_description)
        logger.info(f"🟢 Task sent to Celery")
    except Exception as e:
        logger.error(f"Celery Error: {e}")
        return {"error": "Failed to queue task"}

    return {"session_id": session_id, "status": "processing"}


@app.get("/status/{session_id}", dependencies=[Depends(rate_limiter(limit=60, window_seconds=60))])
async def get_status(session_id: str):
    """Return the current status and result for a screening job.

    A passive read of the DB row written by the worker. Includes a watchdog: a
    job stuck in ``processing`` past a threshold (worker likely died) is reported
    as ``failed`` so the client doesn't poll forever.
    """
    db = SessionLocal()
    try:
        result = db.query(AnalysisResult).filter(AnalysisResult.session_id == session_id).first()
        if not result:
            return {"error": "Not found"}

        # Watchdog: a task stuck "processing" past this threshold means the
        # worker likely crashed/died without ever updating the row. Report it
        # as failed instead of leaving the frontend polling forever.
        status = result.status
        formatted_text = result.result_text
        if status == "processing" and datetime.utcnow() - result.updated_at > timedelta(minutes=3):
            status = "failed"
            formatted_text = formatted_text or "Analysis timed out"

        return {
            "status": status,
            "result": result.result_metadata,
            "formatted_text": formatted_text 
        }
    finally:
        db.close()


def build_followup_context(job_description: str, resume_text: str | None, evaluation: dict | None) -> str:
    """Flatten the stored job description, resume text, and evaluation into a
    plain-text block the follow-up agent can reason over without any tool calls."""
    evaluation = evaluation or {}
    lines = ["JOB DESCRIPTION:", job_description or "(not available)", ""]
    if resume_text:
        lines += ["CANDIDATE RESUME:", resume_text, ""]
    lines.append("PRIOR EVALUATION OF THE CANDIDATE:")
    if "candidate_name" in evaluation:
        lines.append(f"Candidate: {evaluation.get('candidate_name')}")
        lines.append(f"Score: {evaluation.get('score')}")
        lines.append(f"Recommendation: {evaluation.get('final_recommendation')}")
        lines.append(f"Reasoning: {evaluation.get('reasoning')}")
        strengths = evaluation.get("key_strengths") or []
        concerns = evaluation.get("concerns") or []
        if strengths:
            lines.append("Key strengths:")
            lines.extend(f"- {s}" for s in strengths)
        if concerns:
            lines.append("Concerns:")
            lines.extend(f"- {c}" for c in concerns)
    else:
        # Parsing failed earlier — fall back to whatever raw text we stored.
        lines.append(str(evaluation))
    return "\n".join(lines)


@app.post("/sessions/{session_id}/followup", dependencies=[Depends(rate_limiter(limit=20, window_seconds=60))])
async def followup_question(session_id: str, payload: FollowupRequest):
    """Answer a follow-up question about a completed analysis, streamed over SSE.

    Builds context from the stored job description + resume + evaluation and feeds
    it to a tool-free Groq agent, streaming the answer token-by-token as
    ``data: {"delta": ...}`` events ending with ``data: {"done": true}``.
    """
    # Guard: only allow follow-ups on analyses that actually finished —
    # asking about a resume still processing (or one that never existed)
    # would just hallucinate without the original context.
    db = SessionLocal()
    try:
        result = db.query(AnalysisResult).filter(AnalysisResult.session_id == session_id).first()
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        if result.status != "completed":
            raise HTTPException(status_code=400, detail="Analysis is not complete yet")
        context = build_followup_context(result.job_description, result.resume_text, result.result_metadata)
    finally:
        db.close()

    async def event_stream():
        try:
            # Standalone tool-free agent (no Team, no delegation). The original
            # evaluation is passed inline as context, so it answers directly —
            # no `delegate_task_to_member` tool call for Groq to mangle, which
            # also means real token-by-token streaming works here.
            prompt = f"{context}\n\nRecruiter's follow-up question: {payload.question}"
            async for event in followup_agent.arun(prompt, stream=True):
                if event.event == RunEvent.run_content.value and event.content:
                    yield f"data: {json.dumps({'delta': str(event.content)})}\n\n"
                elif event.event == RunEvent.run_error.value:
                    yield f"data: {json.dumps({'error': str(event.content or 'AI request failed')})}\n\n"
                    return
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            logger.error(f"Followup error for {session_id}: {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)