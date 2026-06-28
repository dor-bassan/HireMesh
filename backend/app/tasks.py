"""Celery worker — the asynchronous side of the system.

The API enqueues a task here (via Redis) for every uploaded resume and returns
immediately. This worker, running as a separate process, pulls the task, runs
the multi-agent AI analysis (which is slow), and writes the result back to
PostgreSQL where the API can read it. The worker and the API never talk
directly — Redis carries the task, the database carries the result.
"""
import os
import re
import logging
import json
from celery import Celery
from dotenv import load_dotenv
from app.agents import get_hr_team
from app.database import SessionLocal, AnalysisResult
from app.schemas import CandidateEvaluation

load_dotenv()

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connection to Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("HireMesh", broker=REDIS_URL)

# Worker tuning. Each setting is a deliberate trade-off:
celery_app.conf.update(
    worker_prefetch_multiplier=1,      # grab one task at a time (each is slow & LLM-bound)
    task_acks_late=True,               # ack only after completion -> crash = redelivery, not loss
    worker_pool="solo",                # billiard semaphores fail on Windows/OneDrive; solo avoids multiprocessing
    worker_hijack_root_logger=False,   # prevent Celery from suppressing app-level logs
)


# The task receives the already-extracted resume_text (not the file): text
# extraction happens in the API, so the worker only does the AI work.
@celery_app.task(name="process_resume_analysis", bind=True, max_retries=1)
def process_resume_analysis(self, session_id: str, resume_text: str, job_description: str):
    """Run the full AI screening for one uploaded resume.

    Pulled from Redis by the worker. Steps:
      1. Mark the DB row ``processing`` (skip entirely if already ``completed`` —
         a redelivered task must not redo the expensive LLM work).
      2. Run the multi-agent team on the job description + resume.
      3. Parse the model output into the structured evaluation, save it, and
         mark the row ``completed``.

    On failure the row is marked ``failed``; transient quota errors (429/503)
    are not retried (it would just burn more quota), other errors retry once.

    Args:
        session_id: UUID shared with the DB row and the polling client.
        resume_text: Plain text already extracted from the upload by the API.
        job_description: The role text to evaluate the candidate against.
    """
    print(f"🚀 TASK RECEIVED: {session_id}", flush=True)
    logger.info(f"🚀 Started Task for Session ID: {session_id}")

    db = SessionLocal()

    try:
        # Check that the text is not empty
        if not resume_text:
            logger.error("Resume text is empty!")
            return {"error": "Resume text is empty"}

        # Fetch the row the API created at upload time. If it somehow doesn't
        # exist, the `if result_record` guards below simply skip the updates.
        result_record = db.query(AnalysisResult).filter(AnalysisResult.session_id == session_id).first()

        # Idempotency guard: if a crash happened after commit but before the
        # broker received the ack, this task gets redelivered. Skip redoing
        # the (expensive) AI call if the result is already saved.
        if result_record and result_record.status == "completed":
            logger.info(f"⏭️ Skipping already-completed session: {session_id}")
            return result_record.result_metadata

        if result_record:
            result_record.status = "processing"
            db.commit()

        # 3. Running the AI Team
        hr_team = get_hr_team(session_id=session_id)

        prompt = (
            f"Here is the Job Description:\n{job_description}\n\n"
            f"Here is the Candidate's Resume Content:\n{resume_text}\n\n"
            f"Analyze and provide the structured evaluation."
        )

        logger.info("Sending prompt to AI...")
        response = hr_team.run(prompt)
        ai_output = response.content

        final_json_dict = {}

        # Normalise the AI output into a plain dict. Depending on the model/agno,
        # ``ai_output`` may already be the typed schema, a dict, or a raw string.
        if isinstance(ai_output, CandidateEvaluation):
            final_json_dict = ai_output.model_dump()
        elif isinstance(ai_output, dict):
            final_json_dict = ai_output
        else:
            try:
                raw = str(ai_output).strip()
                # Models often wrap JSON in a ```json ... ``` markdown fence.
                # Strip the opening fence line and the closing fence, then parse.
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)  # drop ```json / ```
                    raw = re.sub(r"\s*```$", "", raw)            # drop trailing ```
                    raw = raw.strip()
                final_json_dict = json.loads(raw)
            except:
                final_json_dict = {"error": "Parsing failed", "raw_content": str(ai_output)}

        # 4. Saving to database
        # Refresh the connection to ensure there are no conflicts
        db.expire_all()
        result_record = db.query(AnalysisResult).filter(AnalysisResult.session_id == session_id).first()

        if result_record:
            result_record.status = "completed"
            result_record.result_metadata = final_json_dict

            # Creating backup text
            if "candidate_name" in final_json_dict:
                strengths = "\n".join([f"- {s}" for s in final_json_dict.get('key_strengths', [])])
                concerns = "\n".join([f"- {s}" for s in final_json_dict.get('concerns', [])])
                result_record.result_text = f"Score: {final_json_dict.get('score')}\nStrengths:\n{strengths}"
            else:
                result_record.result_text = str(ai_output)

            db.commit()
            logger.info(f"✅ SUCCESS: Saved data for {session_id}")

        return final_json_dict

    except Exception as exc:
        logger.error(f"❌ Error: {exc}")
        if 'db' in locals():
            db.rollback()
            try:
                err_record = db.query(AnalysisResult).filter(AnalysisResult.session_id == session_id).first()
                if err_record:
                    err_record.status = "failed"
                    err_record.result_text = str(exc)
                    db.commit()
            except:
                pass
        # Don't retry on 503 (model overloaded) — retrying immediately wastes quota
        if any(code in str(exc) for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
            raise exc
        raise self.retry(exc=exc, countdown=30)
    finally:
        if 'db' in locals():
            db.close()