"""Job scraper module for CareerAgent-RAG.

Provides two scraping backends:
  1. JSearch API (RapidAPI) — primary
  2. Indeed via httpx + BeautifulSoup — fallback

A unified `scrape_jobs` entry-point tries the primary backend first and
falls back to the secondary. `scrape_and_ingest` writes scraped jobs to a
timestamped JSON file and loads them into the ChromaDB vector store.

Usage (CLI)::

    python core/scraper.py --query "ML Engineer" --location "San Francisco" --num 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import List

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ---------------------------------------------------------------------------
# Skill vocabulary
# ---------------------------------------------------------------------------
SKILL_VOCABULARY: List[str] = [
    "Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "C++",
    "React", "Vue", "Angular", "Node.js", "FastAPI", "Django", "Flask",
    "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis",
    "Docker", "Kubernetes", "Terraform",
    "AWS", "GCP", "Azure",
    "Spark", "Kafka", "Airflow", "dbt",
    "PyTorch", "TensorFlow", "scikit-learn", "Pandas", "NumPy",
    "LLMs", "NLP", "MLflow",
    "GraphQL", "REST", "gRPC",
]

_STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "have", "from",
    "not", "but", "they", "will", "been", "has", "more", "also", "than",
    "your", "our", "you", "all", "can", "its", "job", "work", "role",
    "team", "must", "use", "using", "able", "well", "new", "into",
}


# ---------------------------------------------------------------------------
# Skill extraction helper
# ---------------------------------------------------------------------------
def extract_skills_from_text(text: str) -> List[str]:
    """Extract recognised tech skills from *text*.

    Matches against a predefined vocabulary of ~40 common skills
    (case-insensitive). If no vocabulary match is found, returns the top
    three capitalised words that are not common stopwords.

    Args:
        text: Free-form text to scan (job description, snippet, etc.).

    Returns:
        A list of skill strings (may be empty).
    """
    found: List[str] = []
    lower = text.lower()
    for skill in SKILL_VOCABULARY:
        if skill.lower() in lower:
            found.append(skill)
    if found:
        return found

    # Fallback: capitalised words that are not stopwords
    caps = re.findall(r'\b[A-Z][a-z]{1,}\b', text)
    filtered = [w for w in caps if w.lower() not in _STOPWORDS]
    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:3]


# ---------------------------------------------------------------------------
# Backend 1 — JSearch (RapidAPI)
# ---------------------------------------------------------------------------
def scrape_jsearch(
    query: str,
    location: str = "United States",
    num_results: int = 10,
) -> List[dict]:
    """Fetch job listings from the JSearch API (RapidAPI).

    Requires the ``JSEARCH_API_KEY`` environment variable to be set.

    Args:
        query: Job search query, e.g. ``"Python engineer"``.
        location: Geographic filter, e.g. ``"New York"``.
        num_results: Maximum number of jobs to return.

    Returns:
        A list of job dicts compatible with the jobs.json schema.
        Returns ``[]`` on any network or API error.
    """
    api_key = os.environ.get("JSEARCH_API_KEY", "")
    if not api_key:
        logger.warning("JSEARCH_API_KEY is not set. Skipping JSearch backend.")
        return []

    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }
    params = {
        "query": f"{query} in {location}",
        "page": "1",
        "num_pages": "1",
    }

    try:
        with httpx.Client(timeout=15) as client:
            response = client.get(url, headers=headers, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        logger.warning("JSearch HTTP error: %s", exc)
        return []
    except Exception as exc:
        logger.warning("JSearch unexpected error: %s", exc)
        return []

    jobs: List[dict] = []
    data = payload.get("data", [])
    for i, item in enumerate(data[:num_results]):
        raw_skills = item.get("job_required_skills")
        if isinstance(raw_skills, list) and raw_skills:
            skills = [str(s) for s in raw_skills]
        else:
            desc = item.get("job_description", "")
            skills = extract_skills_from_text(desc)

        raw_salary = item.get("job_min_salary")
        min_salary = int(raw_salary) if isinstance(raw_salary, (int, float)) else 0

        jobs.append({
            "job_id": item.get("job_id") or f"jsearch_{i}",
            "title": item.get("job_title", "Unknown Title"),
            "company": item.get("employer_name", "Unknown Company"),
            "description": (item.get("job_description") or "")[:1000],
            "skills_req": skills,
            "min_salary": min_salary,
        })

    logger.info("JSearch returned %d jobs for query '%s'.", len(jobs), query)
    return jobs


# ---------------------------------------------------------------------------
# Backend 2 — Indeed (BeautifulSoup fallback)
# ---------------------------------------------------------------------------
def scrape_indeed(
    query: str,
    location: str = "United States",
    num_results: int = 10,
) -> List[dict]:
    """Scrape job listings from Indeed using httpx and BeautifulSoup.

    Indeed actively blocks automated requests; this function sets a realistic
    User-Agent and waits briefly between requests. If blocked (HTTP 403) it
    logs a warning and returns ``[]``.

    Args:
        query: Job search query, e.g. ``"Python engineer"``.
        location: Geographic filter, e.g. ``"New York"``.
        num_results: Maximum number of jobs to return.

    Returns:
        A list of job dicts compatible with the jobs.json schema.
        Returns ``[]`` on any network or parsing error.
    """
    url = "https://www.indeed.com/jobs"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    params = {"q": query, "l": location, "limit": "15"}

    try:
        time.sleep(1)  # polite delay
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            response = client.get(url, headers=headers, params=params)

        if response.status_code == 403:
            logger.warning(
                "Indeed returned HTTP 403 (bot-blocked). Returning empty list."
            )
            return []

        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Indeed HTTP error: %s", exc)
        return []
    except Exception as exc:
        logger.warning("Indeed unexpected error: %s", exc)
        return []

    soup = BeautifulSoup(response.text, "lxml")

    # Indeed renders different card structures; try both selectors.
    cards = soup.select('[data-testid="slider_item"]')
    if not cards:
        cards = soup.find_all("div", class_=re.compile(r"job_seen_beacon"))

    jobs: List[dict] = []
    for i, card in enumerate(cards[:num_results]):
        # job_id
        job_id = card.get("data-jk") or f"indeed_{i}"

        # title
        title_el = card.select_one('[data-testid="jobTitle"] span') or card.select_one(".jobTitle span")
        title = title_el.get_text(strip=True) if title_el else "Unknown Title"

        # company
        company_el = card.select_one('[data-testid="company-name"]') or card.select_one(".companyName")
        company = company_el.get_text(strip=True) if company_el else "Unknown Company"

        # description snippet
        desc_el = (
            card.select_one(".job-snippet")
            or card.select_one(".underShelfFooter")
        )
        description = desc_el.get_text(" ", strip=True) if desc_el else ""

        skills = extract_skills_from_text(description)

        jobs.append({
            "job_id": job_id,
            "title": title,
            "company": company,
            "description": description[:1000],
            "skills_req": skills,
            "min_salary": 0,
        })

    logger.info("Indeed scraper returned %d jobs for query '%s'.", len(jobs), query)
    return jobs


# ---------------------------------------------------------------------------
# Unified entry-point
# ---------------------------------------------------------------------------
def scrape_jobs(
    query: str,
    location: str = "United States",
    num_results: int = 10,
    backend: str = "auto",
) -> List[dict]:
    """Scrape job listings using the specified or best-available backend.

    Backend resolution for ``"auto"``:
      1. Try JSearch if ``JSEARCH_API_KEY`` is set in the environment.
      2. Fall back to Indeed.

    Results are deduplicated by ``job_id`` and capped at *num_results*.

    Args:
        query: Job search query string.
        location: Geographic filter.
        num_results: Maximum number of jobs to return.
        backend: One of ``"jsearch"``, ``"indeed"``, or ``"auto"``.

    Returns:
        A deduplicated list of job dicts (up to *num_results* items).
    """
    raw: List[dict] = []

    if backend == "jsearch":
        raw = scrape_jsearch(query, location, num_results)
    elif backend == "indeed":
        raw = scrape_indeed(query, location, num_results)
    else:  # auto
        if os.environ.get("JSEARCH_API_KEY"):
            raw = scrape_jsearch(query, location, num_results)
        if not raw:
            logger.info("JSearch yielded no results; trying Indeed fallback.")
            raw = scrape_indeed(query, location, num_results)

    # Deduplicate by job_id
    seen: set = set()
    unique: List[dict] = []
    for job in raw:
        jid = job.get("job_id")
        if jid not in seen:
            seen.add(jid)
            unique.append(job)

    return unique[:num_results]


# ---------------------------------------------------------------------------
# Direct ingest (bypass count-guard in retrieval.py)
# ---------------------------------------------------------------------------
def ingest_jobs_direct(jobs: List[dict]) -> int:
    """Add a list of job dicts directly to the ChromaDB vector store.

    Unlike ``retrieval.ingest_jobs``, this function does NOT skip ingestion
    when the collection is already populated, making it suitable for
    incremental updates. It deduplicates against existing records by
    ``job_id`` before inserting.

    Args:
        jobs: List of job dicts following the jobs.json schema.

    Returns:
        The number of newly added documents.
    """
    from langchain_core.documents import Document
    from core.retrieval import chroma_client, vectorstore

    # Fetch existing job_ids to avoid duplicates
    try:
        collection = chroma_client.get_collection("jobs")
        existing_meta = collection.get(include=["metadatas"]).get("metadatas", [])
        existing_ids: set = {m.get("job_id") for m in existing_meta if m}
    except Exception:
        existing_ids = set()

    documents: List[Document] = []
    for job in jobs:
        if job.get("job_id") in existing_ids:
            logger.debug("Skipping duplicate job_id: %s", job.get("job_id"))
            continue

        page_content = (
            f"{job['title']} at {job['company']}. "
            f"{job['description']} "
            f"Required skills: {', '.join(job['skills_req'])}."
        )
        metadata = {
            "job_id": job["job_id"],
            "title": job["title"],
            "company": job["company"],
            "description": job["description"],
            "skills_req": ", ".join(job["skills_req"]),
            "min_salary": int(job["min_salary"]),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    if not documents:
        logger.info("No new jobs to ingest (all were duplicates).")
        return 0

    vectorstore.add_documents(documents)
    logger.info("Directly ingested %d new jobs into ChromaDB.", len(documents))
    return len(documents)


# ---------------------------------------------------------------------------
# scrape_and_ingest
# ---------------------------------------------------------------------------
def scrape_and_ingest(
    query: str,
    location: str = "United States",
    num_results: int = 10,
    backend: str = "auto",
) -> int:
    """Scrape real job postings and ingest them into the vector store.

    Writes the scraped jobs to a timestamped JSON file under ``data/`` for
    auditability, then calls :func:`ingest_jobs_direct` for incremental
    insertion (no count-guard).

    Args:
        query: Job search query string.
        location: Geographic filter.
        num_results: Maximum number of jobs to return.
        backend: Scraping backend: ``"jsearch"``, ``"indeed"``, or ``"auto"``.

    Returns:
        The number of newly ingested jobs.
    """
    jobs = scrape_jobs(query, location, num_results, backend)
    if not jobs:
        logger.warning("scrape_jobs returned no results. Nothing ingested.")
        return 0

    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(DATA_DIR, f"scraped_jobs_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(jobs, fh, indent=2)
    logger.info("Wrote %d scraped jobs to %s", len(jobs), out_path)

    return ingest_jobs_direct(jobs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Scrape and ingest real job listings.")
    parser.add_argument("--query", default="Software Engineer", help="Job search query")
    parser.add_argument("--location", default="United States", help="Location filter")
    parser.add_argument("--num", type=int, default=10, help="Number of jobs to fetch")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "jsearch", "indeed"],
        help="Scraping backend to use",
    )
    args = parser.parse_args()

    count = scrape_and_ingest(
        query=args.query,
        location=args.location,
        num_results=args.num,
        backend=args.backend,
    )
    print(f"Ingested {count} new jobs into ChromaDB.")
