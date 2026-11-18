# CareerAgent-RAG

> Context-aware AI job matching system — paste your resume, get ranked job matches with reasoning.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange)
![Groq](https://img.shields.io/badge/Groq-LLM-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What it does

- **Semantic job matching**: Embeds your resume and finds the most relevant jobs using ChromaDB vector search (HuggingFace `all-MiniLM-L6-v2` embeddings).
- **LLM-powered reasoning**: A LangChain agent backed by Groq (Llama 3 / Mixtral) scores each match, identifies matched and missing skills, and explains the reasoning.
- **Structured output**: Every response is a validated Pydantic `JobMatch` schema — no hallucinated fields.
- **Real job scraping**: Optionally scrape live job listings via JSearch API and ingest them into the vector store.

---

## Architecture

```
Resume Text (plain text)
        │
        ▼
  ┌─────────────┐
  │   FastAPI   │  POST /match-jobs
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │  LangChain Agent │  (tool-calling loop)
  └──────┬───────────┘
         │
    ┌────┴────────────────────┐
    │                         │
    ▼                         ▼
┌──────────┐         ┌──────────────┐
│ Groq LLM │         │   ChromaDB   │
│ (Llama3/ │         │ Vector Store │
│ Mixtral) │         │  (HuggingFace│
└──────────┘         │  embeddings) │
                     └──────────────┘
         │
         ▼
  JobMatch JSON  (job_id, match_score, matched_skills,
                  missing_skills, reasoning)
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Backend API | FastAPI | REST endpoints, request validation |
| LLM Orchestration | LangChain | Agent loop, tool calling |
| LLM Inference | Groq (Llama 3 8B / Mixtral 8x7B) | Fast, free-tier LLM inference |
| Vector Database | ChromaDB | Persistent semantic job storage |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Sentence embeddings |
| Validation | Pydantic v2 | Strict structured output schemas |
| Job Scraping | JSearch API (via RapidAPI) | Live job listings |
| Evaluation | Ragas | RAG pipeline quality metrics |

---

## API Endpoints

| Method | Endpoint | Description | Request Body |
|---|---|---|---|
| `GET` | `/` | Health check / welcome message | — |
| `GET` | `/health` | Liveness probe for deployment platforms | — |
| `POST` | `/match-jobs` | Match a resume to jobs | `{"resume_text": "I am a Python developer with 3 years…"}` |
| `POST` | `/ingest-jobs` | Ingest (or re-ingest) `jobs.json` into ChromaDB | — |
| `POST` | `/scrape-jobs` | Scrape live jobs via JSearch and ingest them | `{"query": "Python developer", "num_pages": 2}` |

Interactive docs: `http://localhost:8000/docs`

---

## Quick Start (Local)

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/careeragent-rag.git
   cd careeragent-rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and fill in your API keys
   ```

   Required keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   JSEARCH_API_KEY=your_rapidapi_key_here   # optional, for live scraping
   ```

4. **Start the server**
   ```bash
   uvicorn api:app --reload
   ```

5. **Open the interactive API docs**
   ```
   http://localhost:8000/docs
   ```

   POST to `/ingest-jobs` first to populate the vector store, then POST your resume text to `/match-jobs`.

---

## Deploy to Railway

Railway is the easiest free-tier option. The `railway.toml` and `Dockerfile` are already configured.

1. Push your code to a GitHub repository.
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Select your repository.
4. In the Railway dashboard → **Variables**, add:
   - `GROQ_API_KEY` = your Groq API key
   - `JSEARCH_API_KEY` = your RapidAPI key (optional)
5. Railway auto-detects the `Dockerfile` and deploys. Your API is live at the Railway-generated URL.

> **Note on storage**: Railway's free tier has ephemeral storage. ChromaDB data is re-ingested from `jobs.json` on every cold start. For persistent vector storage across deploys, enable a **Railway Volume** mounted at `/app/vector_store`, or migrate to a managed vector DB (Pinecone, Weaviate).

---

## Deploy to Render

1. Push your code to a GitHub repository.
2. Go to [render.com](https://render.com) → **New** → **Web Service** → connect your repo.
3. Render reads `render.yaml` automatically and configures the service.
4. In the Render dashboard → **Environment**, add:
   - `GROQ_API_KEY`
   - `JSEARCH_API_KEY` (optional)
5. Click **Deploy**. Your service spins up at a `*.onrender.com` URL.

> **Note on storage**: Render's free tier also uses ephemeral storage. Same caveat as Railway above.

---

## Deploy with Docker

**Build and run locally:**
```bash
# Build
docker build -t careeragent-rag .

# Run (pass env vars at runtime, never bake them into the image)
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key_here \
  -e JSEARCH_API_KEY=your_key_here \
  careeragent-rag
```

**Push to any cloud registry:**
```bash
# Docker Hub
docker tag careeragent-rag your-dockerhub-username/careeragent-rag:latest
docker push your-dockerhub-username/careeragent-rag:latest

# Google Artifact Registry / AWS ECR / Azure ACR — same pattern:
docker tag careeragent-rag <registry-url>/careeragent-rag:latest
docker push <registry-url>/careeragent-rag:latest
```

Then deploy the image on any container platform: Google Cloud Run, AWS ECS/Fargate, Azure Container Apps, Fly.io, etc.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Groq API key for LLM inference. Get one free at [console.groq.com](https://console.groq.com). |
| `JSEARCH_API_KEY` | ⚠️ Optional | RapidAPI key for JSearch (live job scraping). Only needed for `/scrape-jobs`. Get it at [rapidapi.com/jsearch](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch). |
| `PORT` | Auto-set | Set automatically by Railway/Render. The `startCommand` reads `$PORT`. Do not set manually. |

---

## Project Structure

```
careeragent-rag/
├── api.py                  # FastAPI app — entry point
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container image definition
├── .dockerignore           # Files excluded from Docker build context
├── railway.toml            # Railway deployment config
├── render.yaml             # Render Blueprint config
├── Procfile                # Heroku/Railway fallback process file
├── .gitignore              # Git ignore rules
├── .env.example            # Environment variable template
├── README.md               # This file
├── LICENSE                 # MIT License
├── core/
│   ├── __init__.py
│   ├── schemas.py          # Pydantic models (JobMatch, ResumeUpload)
│   ├── retrieval.py        # ChromaDB ingest + semantic search
│   ├── agent.py            # LangChain tool-calling agent
│   └── scraper.py          # JSearch API scraper
├── evals/
│   ├── __init__.py
│   └── evaluate_rag.py     # Ragas evaluation pipeline
├── data/
│   └── jobs.json           # Mock job descriptions (10 entries)
└── vector_store/           # ChromaDB persistent storage (gitignored)
```

---

## Evaluation

Run the Ragas evaluation pipeline against mock data:

```bash
python evals/evaluate_rag.py
```

This script:
1. Sends 3 mock resume queries through the full retrieval pipeline.
2. Collects the retrieved job contexts and LLM-generated answers.
3. Computes two Ragas metrics:
   - **`context_precision`**: Did the vector search retrieve the right jobs? (Higher = retrieval is accurate.)
   - **`faithfulness`**: Did the LLM stick to the retrieved context, or hallucinate skills/details? (Higher = less hallucination.)
4. Prints a summary table to the console.

Expected output:
```
Evaluation Results:
  context_precision: 0.87
  faithfulness:      0.92
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
