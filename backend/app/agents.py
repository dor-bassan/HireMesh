"""AI agent layer.

Defines the agents and models used for analysis and follow-up Q&A, wires up
Langfuse (managed prompts + OpenTelemetry tracing), and exposes a factory for
the analysis team.

Two distinct AI paths live here, on purpose:
  * Analysis  -> a multi-agent **Team** (coordinator + resume parser + job
    analyst) on **Gemini**, because the coordinator delegates via function/tool
    calls and Gemini's tool-calling is reliable.
  * Follow-up -> a single tool-free **agent** on **Groq**, because it just
    answers from context (no delegation), so it's reliable and can stream.

NOTE: prompts are fetched from Langfuse at import time with no fallback — if
Langfuse is unreachable on startup, importing this module raises and the process
won't boot. That's deliberate: never run with stale/wrong prompts.
"""
import os
import base64
import logging
from dotenv import load_dotenv
from langfuse import Langfuse  # Import the Langfuse SDK

# Agno Imports
from agno.agent import Agent
from agno.team import Team
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.db.postgres import PostgresDb

# Local Imports
from app.database import DATABASE_URL

# OpenTelemetry / Tracing Imports
from openinference.instrumentation.agno import AgnoInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

load_dotenv()

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Langfuse & Tracing Configuration ---
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_host = os.getenv("LANGFUSE_BASE_URL")

if not public_key or not secret_key:
    logger.warning("⚠️ Warning: Langfuse keys missing. Tracing skipped.")
else:
    # 1. Create Auth Header for OTEL
    LANGFUSE_AUTH = base64.b64encode(
        f"{public_key}:{secret_key}".encode()
    ).decode()

    # 2. Configure OTEL Endpoint
    otel_endpoint = f"{langfuse_host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_endpoint
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

    # 3. Setup Tracer Provider
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    trace_api.set_tracer_provider(tracer_provider=tracer_provider)

    # 4. Instrument Agno
    AgnoInstrumentor().instrument()

    # The OTEL instrumentation logs a benign "Token was created in a different
    # Context" error when a tracing span's context is attached and detached
    # across async-generator / worker-thread boundaries (e.g. the streaming
    # follow-up). The traced work itself completes fine — this only quiets the
    # non-actionable detach noise so the console stays readable.
    logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

    logger.info(f"✅ Langfuse Tracing Enabled on host: {langfuse_host}")

# --- Langfuse Client Initialization ---
langfuse = Langfuse()


def get_prompt_content(prompt_name: str) -> str:
    """
    Strict function to fetch a prompt from Langfuse.
    NO FALLBACK: If fetching fails, this will raise an exception and stop the worker.
    This ensures we only ever use the managed prompts.
    """
    try:
        # Fetch the production version of the prompt
        prompt = langfuse.get_prompt(prompt_name)

        # Log success for verification
        logger.info(f"✨ Successfully loaded prompt '{prompt_name}' from Langfuse")

        # Compile returns the final string
        return prompt.compile()
    except Exception as e:
        # Critical error logging before crashing
        logger.critical(f"❌ CRITICAL ERROR: Failed to fetch prompt '{prompt_name}' from Langfuse.")
        logger.critical("Check your API Keys and Prompt Names in Langfuse Dashboard.")
        raise e  # Re-raise the exception to crash the task/worker


print("🚀 Starting Agent with Langfuse Tracking & Strict Prompt Management...")

# --- Database for Agent Sessions ---
agent_db = PostgresDb(
    db_url=DATABASE_URL,
    session_table="agent_sessions",
)

# --- Model Configuration ---
# Analysis (the Team) runs on Gemini: its delegation is done via tool calls,
# and Gemini's function-calling is reliable, whereas Groq's llama-3.3
# intermittently emits a malformed `delegate_task_to_member` call.
model_fast = Gemini(id="gemini-3.5-flash")
model_reasoning = Gemini(id="gemini-3.5-flash")

# Follow-up runs on Groq: it's a single tool-free agent (no delegation), so
# Groq is reliable here — and keeping it off Gemini saves that quota.
model_followup = Groq(id="llama-3.3-70b-versatile")

# --- Agents Configuration ---

# 1. Resume Parser Agent
resume_instructions = get_prompt_content("resume-parser-instructions")

resume_parser = Agent(
    id="resume-parser",
    name="Resume Parser",
    role="Extract details from candidate resumes",
    model=model_fast,
    instructions=[resume_instructions],
)

# 2. Job Analyst Agent
job_instructions = get_prompt_content("job-analyst-instructions")

job_analyst = Agent(
    id="job-analyst",
    name="Job Analyst",
    role="Analyze job descriptions",
    model=model_fast,
    instructions=[job_instructions],
)


# 3. Follow-up Assistant Agent
#
# Deliberately a standalone Agent with NO members and NO tools — unlike the
# Team, it never makes a delegation tool call. Groq's llama-3.3 reliably
# botches the `delegate_task_to_member` tool call on follow-up turns, so the
# follow-up path answers directly from context passed in the prompt instead.
# Tool-free also means Groq streaming works, so this path can stream for real.
followup_agent = Agent(
    id="followup-assistant",
    name="Follow-up Assistant",
    role="Answer recruiter follow-up questions about an already-evaluated candidate",
    model=model_followup,
    instructions=[
        "A candidate has already been evaluated against a job description. "
        "You are given that job description and the structured evaluation as context.",
        "Answer the recruiter's follow-up question directly and concisely, grounded only in that context.",
        "If the context does not contain enough information to answer, say so plainly instead of inventing details.",
    ],
)


# --- Team Configuration ---

def get_hr_team(session_id: str):
    """Build the analysis Team for one screening job.

    Re-created per request (and re-fetches the team-lead prompt each time) so it
    can be bound to this ``session_id`` for Agno's session memory. The coordinator
    delegates to ``resume_parser`` and ``job_analyst`` and returns the final
    structured evaluation.
    """
    team_lead_instructions = get_prompt_content("hr-team-lead-instructions")

    return Team(
        name="HR Recruitment Team",
        members=[resume_parser, job_analyst],
        model=model_reasoning,
        db=agent_db,
        session_id=session_id,
        instructions=[team_lead_instructions],
    )