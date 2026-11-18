"""LangChain tool-calling agent for CareerAgent-RAG.

Wraps ChromaDB semantic search and salary filtering as LangChain tools,
orchestrates them via a Groq LLM, and returns structured JobMatch results.
"""
import json
import logging
import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

from core.retrieval import get_jobs_by_min_salary, semantic_job_search
from core.schemas import JobMatch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def search_jobs(query: str) -> str:
    """Search for jobs matching a query string. Returns a JSON string of matching job metadata."""
    results = semantic_job_search(query, top_k=5)
    return json.dumps(results, indent=2)


@tool
def filter_jobs_by_salary(min_salary: int) -> str:
    """Filter jobs by minimum salary requirement. Returns a JSON string of jobs meeting the salary threshold."""
    results = get_jobs_by_min_salary(min_salary)
    return json.dumps(results, indent=2)


tools = [search_jobs, filter_jobs_by_salary]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert career advisor and job matching agent. 
Given a candidate's resume text, your task is to:
1. Analyse the resume to identify key skills, experience, and preferences.
2. Use the search_jobs tool to find semantically relevant job postings.
3. Optionally use the filter_jobs_by_salary tool if a salary preference is mentioned.
4. For each promising job found, reason about the match quality, matched skills, and missing skills.
5. Return a JSON array of up to 5 job match objects. Each object must have exactly these fields:
   - job_id: string
   - match_score: integer 1-100
   - matched_skills: list of strings
   - missing_skills: list of strings
   - reasoning: string explaining the match
Return ONLY the JSON array, no other text.""",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# ---------------------------------------------------------------------------
# Agent executor
# ---------------------------------------------------------------------------
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def run_agent(resume_text: str) -> List[JobMatch]:
    """Run the job-matching agent for a given resume.

    The agent searches ChromaDB for relevant jobs, reasons about each match,
    and returns a structured list of up to 5 JobMatch objects.

    Args:
        resume_text: Raw text content of the candidate's resume.

    Returns:
        A list of JobMatch objects ordered by relevance. Returns an empty
        list if the agent fails or produces unparseable output.
    """
    try:
        result = agent_executor.invoke({"input": resume_text})
        raw_output: str = result.get("output", "")

        # --- Attempt 1: parse as a JSON array ---
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, list):
                matches: List[JobMatch] = []
                for item in parsed:
                    try:
                        matches.append(JobMatch(**item))
                    except Exception as validation_err:
                        logger.warning("Skipping invalid job match item: %s", validation_err)
                return matches[:5]
        except json.JSONDecodeError:
            logger.warning("Agent output is not valid JSON. Attempting structured extraction.")

        # --- Attempt 2: use structured output to rescue a single match ---
        structured_llm = llm.with_structured_output(JobMatch)
        single_match: JobMatch = structured_llm.invoke(
            f"Extract job match details from: {raw_output}"
        )
        return [single_match]

    except Exception as exc:
        logger.error("Agent execution failed: %s", exc, exc_info=True)
        return []
