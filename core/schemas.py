"""Pydantic schemas for CareerAgent-RAG structured outputs."""
from typing import List

from pydantic import BaseModel, Field


class JobMatch(BaseModel):
    """Represents a single job match result from the agent."""

    job_id: str = Field(..., description="Unique identifier of the matched job")
    match_score: int = Field(..., ge=1, le=100, description="Match score from 1 to 100")
    matched_skills: List[str] = Field(
        default_factory=list,
        description="Skills in the resume that match the job requirements",
    )
    missing_skills: List[str] = Field(
        default_factory=list,
        description="Skills required by the job that are missing from the resume",
    )
    reasoning: str = Field(..., description="Explanation of why this job matches the resume")


class ResumeUpload(BaseModel):
    """API request body for the job-matching endpoint."""

    resume_text: str = Field(
        ...,
        min_length=50,
        description="Raw text content of the candidate's resume",
    )


class JobMatchResponse(BaseModel):
    """API response body containing a list of job matches."""

    matches: List[JobMatch] = Field(
        default_factory=list,
        description="List of matched jobs",
    )
    total_found: int = Field(..., description="Total number of matches returned")
