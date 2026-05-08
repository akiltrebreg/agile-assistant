"""RAGAS evaluation metrics wrapper.

Metrics
-------
Retrieval quality  : context_precision, context_recall
Generation quality : faithfulness, answer_relevancy
End-to-end         : answer_correctness

Uses GPT-5.2 via vsellm as LLM-as-judge.

Usage::

    from eval.metrics import evaluate_rag, RAGSample

    samples = [
        RAGSample(
            question="Какой целевой порог Done Total?",
            answer="80 %",
            contexts=["Done Total >= 80 % ..."],
            ground_truth="Целевой порог Done Total — 80 %.",
        ),
    ]
    results = evaluate_rag(samples)
    print(results["scores"])
"""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

JUDGE_MODEL = "openai/gpt-5.2"
JUDGE_BASE_URL = "https://api.vsellm.ru/v1"
EMBEDDING_MODEL: str | None = None  # lazy: set from settings in _build_embeddings()

METRIC_NAMES = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
)


@dataclass
class RAGSample:
    """Single RAG evaluation sample."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


# ── builder helpers ──────────────────────────────────────────


def _build_judge_llm():
    """Build LLM-as-judge via vsellm OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=os.environ["VSELLM_API_KEY"],
        base_url=os.environ.get("VSELLM_BASE_URL", JUDGE_BASE_URL),
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def _build_embeddings():
    """Build embeddings for the answer_relevancy metric.

    Uses the same S3-aware resolver as the production retriever
    (``ensure_embedding_model_downloaded``) so the local snapshot under
    ``/app/models/{embedding_model}/`` is reused. ``settings.embedding_model``
    holds a folder name (not a HF Hub ID) when ``S3_MODELS_BUCKET`` is set,
    so passing it directly to ``HuggingFaceEmbeddings`` would force a Hub
    lookup at ``sentence-transformers/<folder>`` and fail with 401.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from agile_assistant.rag.embeddings import ensure_embedding_model_downloaded

    emb = HuggingFaceEmbeddings(
        model_name=ensure_embedding_model_downloaded(),
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(emb)


# ── public API ───────────────────────────────────────────────


def evaluate_rag(samples: list[RAGSample]) -> dict:
    """Run all RAGAS metrics on *samples*.

    Returns::

        {
            "scores": {"faithfulness": 0.85, ...},   # mean per metric
            "per_sample": [{"faithfulness": 0.9, ...}, ...],
        }
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        AnswerCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    llm = _build_judge_llm()
    emb = _build_embeddings()

    metrics = [
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=emb),
        AnswerCorrectness(llm=llm),
    ]

    ragas_samples = [
        SingleTurnSample(
            user_input=s.question,
            response=s.answer,
            retrieved_contexts=s.contexts,
            reference=s.ground_truth,
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=ragas_samples)

    logger.info("Running RAGAS evaluation on %d samples …", len(samples))
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=emb)

    df = result.to_pandas()
    metric_cols = [
        c
        for c in df.columns
        if c not in ("user_input", "response", "retrieved_contexts", "reference")
    ]

    return {
        "scores": {col: float(df[col].mean()) for col in metric_cols},
        "per_sample": df[metric_cols].to_dict(orient="records"),
    }
