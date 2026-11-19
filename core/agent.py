"""LangChain tool-calling agent for CareerAgent-RAG.

Wraps ChromaDB semantic search and salary filtering as LangChain tools,
orchestrates them via a Groq LLM, and returns structured JobMatch results.
Uses the modern LangChain v0.2+ LCEL pipeline (no AgentExecutor).
"""
import json
import logging
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
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
    """Search for jobs semantically matching a query string."""
    results = semantic_job_search(query, top_k=5)
    return json.dumps(results, indent=2)


@tool
def filter_jobs_by_salary(min_salary: int) -> str:
    """Filter jobs by minimum salary. Returns jobs paying at least min_salary USD/year."""
    results = get_jobs_by_min_salary(min_salary)
    return json.dumps(results, indent=2)


tools = [search_jobs, filter_jobs_by_salary]
llm_with_tools = llm.bind_tools(tools)

TOOLS_MAP = {t.name: t for t in tools}

SYSTEM_PROMPT = """You are an expert career advisor and job matching agent.
Given a candidate's resume text:
1. Use the search_jobs tool to find semantically relevant job postings.
2. Optionally use filter_jobs_by_salary if a salary preference is mentioned.
3. After getting results, return ONLY a JSON array of up to 5 job match objects.

Each object must have exactly these fields:
- job_id: string
- match_score: integer 1-100
- matched_skills: list of strings
- missing_skills: list of strings
- reasoning: string

Return ONLY the JSON array, no markdown, no explanation."""


def _run_tool(tool_call: dict) -> str:
    """Execute a single tool call and return its string output."""
    name = tool_call["name"]
    args = tool_call["args"]
    if name not in TOOLS_MAP:
        return f"Unknown tool: {name}"
    return TOOLS_MAP[name].invoke(args)


def run_agent(resume_text: str) -> List[JobMatch]:
    """Run the job-matching agent for a given resume.

    Uses a manual tool-calling loop compatible with LangChain v1.x.

    Args:
        resume_text: Raw text content of the candidate's resume.

    Returns:
        A list of up to 5 JobMatch objects. Returns empty list on failure.
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=resume_text),
    ]

    try:
        # Agentic loop — max 5 iterations
        for _ in range(5):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # If no tool calls, we have the final answer
            if not response.tool_calls:
                break

            # Execute each tool call and feed results back
            for tc in response.tool_calls:
                result = _run_tool(tc)
                from langchain_core.messages import ToolMessage
                messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )

        # Extract final text output
        final_output = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_output = msg.content
                break

        if not final_output:
            logger.warning("Agent produced no text output.")
            return []

        # Strip markdown code fences if present
        final_output = final_output.strip()
        if final_output.startswith("```"):
            lines = final_output.split("\n")
            final_output = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()

        # Parse JSON array
        parsed = json.loads(final_output)
        if not isinstance(parsed, list):
            parsed = [parsed]

        matches: List[JobMatch] = []
        for item in parsed:
            try:
                matches.append(JobMatch(**item))
            except Exception as e:
                logger.warning("Skipping invalid match item: %s", e)

        return matches[:5]

    except json.JSONDecodeError:
        logger.warning("Agent output not valid JSON, trying structured extraction.")
        try:
            structured_llm = llm.with_structured_output(JobMatch)
            match = structured_llm.invoke(
                f"Extract one job match from: {final_output}"
            )
            return [match]
        except Exception as e:
            logger.error("Structured extraction failed: %s", e)
            return []

    except Exception as exc:
        logger.error("Agent execution failed: %s", exc, exc_info=True)
        return []
