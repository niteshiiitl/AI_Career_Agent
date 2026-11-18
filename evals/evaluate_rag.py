"""Standalone RAG evaluation script for CareerAgent-RAG.

Uses the Ragas library to measure context_precision and faithfulness
against a small set of mock resume queries and expected answers.

Run with:
    python evals/evaluate_rag.py
"""
import os
import sys

# Ensure the project root is on the Python path so core.* imports resolve.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

import os  # re-imported after path setup for clarity in later calls

from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, faithfulness

from core.retrieval import semantic_job_search

# ---------------------------------------------------------------------------
# Mock resume queries
# ---------------------------------------------------------------------------
QUERIES = [
    (
        "Python ML engineer with 4 years experience in PyTorch, TensorFlow, MLflow, "
        "and Kubernetes. Seeking senior role with salary above 140000."
    ),
    (
        "Frontend developer skilled in React, TypeScript, and CSS with 3 years building "
        "responsive web applications. Comfortable with Storybook and Webpack."
    ),
    (
        "Cloud and DevOps specialist with expertise in Terraform, AWS, Kubernetes, "
        "Docker, and CI/CD pipelines. 5 years of infrastructure experience."
    ),
]

# ---------------------------------------------------------------------------
# Ground-truth answers (ideal, authoritative)
# ---------------------------------------------------------------------------
GROUND_TRUTHS = [
    (
        "The best matches for this ML engineer are the Machine Learning Engineer role at "
        "NeuralPath AI (job_001) and the AI Research Scientist position at Cognify Labs "
        "(job_009), both requiring Python, PyTorch, and Kubernetes."
    ),
    (
        "The top match for this frontend developer is the Frontend Engineer role at "
        "Pixel & Co. (job_005), which requires React, TypeScript, CSS, Storybook, and Webpack."
    ),
    (
        "The strongest matches for this DevOps specialist are the DevOps Engineer role at "
        "CloudOps Pro (job_004) and the Cloud Architect position at Nimbus Infrastructure "
        "(job_008), both heavily focused on Terraform, AWS, and Kubernetes."
    ),
]

# ---------------------------------------------------------------------------
# Mock LLM-generated answers (slightly imperfect to test faithfulness)
# ---------------------------------------------------------------------------
GENERATED_ANSWERS = [
    (
        "Top matches: Machine Learning Engineer at NeuralPath AI (job_001) — strong overlap "
        "with PyTorch, MLflow, Kubernetes, and Hadoop (candidate has strong Hadoop experience). "
        "Also AI Research Scientist at Cognify Labs (job_009) aligns well with LLM and "
        "PyTorch expertise."
    ),
    (
        "Best match: Frontend Engineer at Pixel & Co. (job_005). The candidate's React, "
        "TypeScript, Storybook, and Angular skills align closely. The role also requires "
        "Vue.js, which the candidate has not listed."
    ),
    (
        "Top matches: DevOps Engineer at CloudOps Pro (job_004) with Terraform, AWS, "
        "Kubernetes overlap, and Cloud Architect at Nimbus Infrastructure (job_008). "
        "The candidate's Ansible expertise is also highly relevant to both roles."
    ),
]


def build_dataset() -> Dataset:
    """Retrieve contexts for each query and package everything into a Ragas Dataset.

    Returns:
        A HuggingFace Dataset with columns: question, answer, contexts, ground_truth.
    """
    all_contexts = []
    for query in QUERIES:
        job_metas = semantic_job_search(query, top_k=3)
        # Build a readable context string per retrieved job.
        contexts = [
            f"{m.get('title', '')} at {m.get('company', '')}. "
            f"{m.get('description', '')} "
            f"Required skills: {m.get('skills_req', '')}."
            for m in job_metas
        ]
        all_contexts.append(contexts)

    return Dataset.from_dict(
        {
            "question": QUERIES,
            "answer": GENERATED_ANSWERS,
            "contexts": all_contexts,
            "ground_truth": GROUND_TRUTHS,
        }
    )


if __name__ == "__main__":
    try:
        print("Building evaluation dataset...")
        dataset = build_dataset()

        print("Configuring Ragas LLM and embeddings (Groq + HuggingFace)...")
        groq_llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        ragas_llm = LangchainLLMWrapper(groq_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

        for metric in [context_precision, faithfulness]:
            metric.llm = ragas_llm
            metric.embeddings = ragas_embeddings

        print("Running Ragas evaluation (context_precision, faithfulness)...")
        result = evaluate(dataset, metrics=[context_precision, faithfulness])

        print("\n=== RAG Evaluation Results ===")
        print(result)

    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        raise
