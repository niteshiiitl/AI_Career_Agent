"""Semantic retrieval module for CareerAgent-RAG.

Handles ChromaDB initialisation, job ingestion, and vector-similarity search.
"""
import json
import logging
import os
from typing import List

import chromadb
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "vector_store")

# ---------------------------------------------------------------------------
# Shared clients (module-level singletons)
# ---------------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    client=chroma_client,
    collection_name="jobs",
    embedding_function=embeddings,
)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def ingest_jobs(json_path: str) -> int:
    """Read jobs from a JSON file and store them in ChromaDB.

    Skips ingestion if the collection already contains documents.

    Args:
        json_path: Absolute path to the jobs JSON file.

    Returns:
        The number of documents ingested (0 if skipped).
    """
    try:
        existing_count = chroma_client.get_collection("jobs").count()
        if existing_count > 0:
            logger.warning(
                "ChromaDB collection 'jobs' already contains %d documents. "
                "Skipping ingestion.",
                existing_count,
            )
            return 0
    except Exception:
        pass

    with open(json_path, "r", encoding="utf-8") as fh:
        jobs = json.load(fh)

    documents: List[Document] = []
    for job in jobs:
        page_content = (
            f"{job['title']} at {job['company']}. "
            f"{job['description']} "
            f"Required skills: {', '.join(job['skills_req'])}."
        )
        metadata = {
            "job_id":      job["job_id"],
            "title":       job["title"],
            "company":     job["company"],
            "description": job["description"],
            "skills_req":  ", ".join(job["skills_req"]),
            "min_salary":  int(job["min_salary"]),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    vectorstore.add_documents(documents)
    logger.info("Ingested %d jobs into ChromaDB.", len(documents))
    return len(documents)


def semantic_job_search(query: str, top_k: int = 5) -> List[dict]:
    """Return the most semantically relevant jobs for a free-text query.

    Args:
        query: Natural-language search query (e.g. a skills summary).
        top_k: Maximum number of results to return.

    Returns:
        A list of job metadata dictionaries ordered by relevance.
    """
    results = vectorstore.similarity_search(query, k=top_k)
    return [doc.metadata for doc in results]


def get_jobs_by_min_salary(min_salary: int) -> List[dict]:
    """Return all jobs whose minimum salary is at least *min_salary*.

    Args:
        min_salary: Lower bound on the annual salary (USD).

    Returns:
        A list of job metadata dictionaries that meet the salary threshold.
    """
    collection = chroma_client.get_collection("jobs")
    response = collection.get(
        where={"min_salary": {"$gte": min_salary}},
        include=["metadatas"],
    )
    return response.get("metadatas", [])


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    jobs_path = os.path.join(PROJECT_ROOT, "data", "jobs.json")
    count = ingest_jobs(jobs_path)
    print(f"Ingested {count} jobs into ChromaDB.")
