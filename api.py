"""FastAPI application for CareerAgent-RAG job matching service."""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

from core.agent import run_agent
from core.retrieval import chroma_client, ingest_jobs
from core.schemas import JobMatchResponse, ResumeUpload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "jobs.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: ingest jobs into ChromaDB on startup if collection is empty."""
    try:
        collection = chroma_client.get_or_create_collection("jobs")
        if collection.count() == 0:
            logger.info("ChromaDB collection is empty. Ingesting jobs...")
            count = ingest_jobs(JOBS_JSON_PATH)
            logger.info("Ingested %d jobs into ChromaDB.", count)
        else:
            logger.info(
                "ChromaDB collection already has %d documents. Skipping ingestion.",
                collection.count(),
            )
    except Exception as exc:
        logger.error("Startup ingestion failed: %s", exc)
    yield


app = FastAPI(
    title="CareerAgent RAG API",
    version="1.0.0",
    description="Context-aware job matching system powered by LangChain, Groq, and ChromaDB.",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/match-jobs", response_model=JobMatchResponse)
async def match_jobs(request: ResumeUpload):
    """Match a candidate's resume against available job postings.

    Accepts the raw text of a resume, runs it through the LangChain agent,
    and returns a ranked list of JobMatch objects.
    """
    logger.info("Received /match-jobs request.")
    try:
        matches = run_agent(request.resume_text)
        return JobMatchResponse(matches=matches, total_found=len(matches))
    except Exception as exc:
        logger.error("Error in /match-jobs: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(exc)}")
