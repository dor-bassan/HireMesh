"""Pydantic models used for AI output and request validation.

``CandidateEvaluation`` is the structured shape the analysis is expected to
produce; ``FollowupRequest`` validates the body of the follow-up endpoint.
"""
from typing import List
from pydantic import BaseModel, Field


class CandidateEvaluation(BaseModel):
    """The structured verdict the AI team returns for one candidate."""

    candidate_name: str = Field(..., description="The full name of the candidate found in the resume.")
    score: int = Field(..., description="A score between 0-100 indicating fit for the job.")

    # Lists allow us to display bullet points in the Frontend
    key_strengths: List[str] = Field(..., description="List of 3-5 major strengths relevant to the job.")
    concerns: List[str] = Field(..., description="List of potential concerns or missing skills.")

    # Verbal summary
    reasoning: str = Field(..., description="A concise summary explaining the score and recommendation.")

    # Clear final recommendation
    final_recommendation: str = Field(..., description="One of: 'Strong Hire', 'Hire', 'Caution', 'Reject'.")


class FollowupRequest(BaseModel):
    """Body of ``POST /sessions/{session_id}/followup`` — a single question."""

    question: str = Field(..., min_length=1, max_length=2000)