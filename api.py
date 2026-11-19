"""FastAPI application for CareerAgent-RAG job matching service."""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

load_dotenv()

import asyncio
from pydantic import BaseModel, Field

from core.agent import run_agent
from core.retrieval import chroma_client, ingest_jobs, VECTOR_STORE_PATH
from core.schemas import JobMatchResponse, ResumeUpload
from core.scraper import scrape_and_ingest

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

# Serve index.html at the root so users get the UI when they visit the Render URL
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend UI."""
    index_path = os.path.join(PROJECT_ROOT, "index.html")
    return FileResponse(index_path, media_type="text/html")


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


class ScrapeRequest(BaseModel):
    """Request body for the /scrape-jobs endpoint."""
    query: str
    location: str = "United States"
    num_results: int = Field(default=10, ge=1, le=50)
    backend: str = "auto"


@app.post("/scrape-jobs")
async def scrape_and_index_jobs(request: ScrapeRequest):
    """Scrape real job postings and add them to the vector index.

    Uses the scraper module to fetch live job listings from JSearch or
    Indeed, deduplicates them against the existing ChromaDB collection,
    and ingests any new entries.
    """
    try:
        count = await asyncio.to_thread(
            scrape_and_ingest,
            request.query,
            request.location,
            request.num_results,
            request.backend,
        )
        return {"message": f"Scraping complete.", "jobs_ingested": count}
    except Exception as exc:
        logger.error("Error in /scrape-jobs: %s", exc)
        raise HTTPException(status_code=500, detail=f"Scraper error: {str(exc)}")


@app.post("/upload-resume", response_model=JobMatchResponse)
async def upload_resume(file: UploadFile = File(...)):
    """Accept a PDF, DOCX, or TXT resume file and return job matches."""
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".pdf", ".docx", ".txt"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: .pdf, .docx, .txt",
        )
    content = await file.read()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"PDF parse error: {exc}")
    elif ext == ".docx":
        try:
            from docx import Document
            import io
            doc = Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"DOCX parse error: {exc}")
    else:
        text = content.decode("utf-8", errors="replace")
    if len(text.strip()) < 50:
        raise HTTPException(
            status_code=422,
            detail="Extracted text is too short (< 50 characters). Please upload a valid resume.",
        )
    logger.info("Received /upload-resume request. Extracted %d chars.", len(text))
    try:
        matches = run_agent(text)
        return JobMatchResponse(matches=matches, total_found=len(matches))
    except Exception as exc:
        logger.error("Error in /upload-resume: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(exc)}")


@app.get("/vector-db/status")
async def vector_db_status():
    """Return live stats about the ChromaDB collection."""
    try:
        collection = chroma_client.get_or_create_collection("jobs")
        total = collection.count()
        return {
            "collection": "jobs",
            "total_documents": total,
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_store_path": VECTOR_STORE_PATH,
            "status": "ready",
        }
    except Exception as exc:
        logger.error("Error in /vector-db/status: %s", exc)
        raise HTTPException(status_code=500, detail=f"ChromaDB error: {str(exc)}")
