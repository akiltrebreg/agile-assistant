"""Count exact tokens in Supervisor's prompt using the real avibe tokenizer.

Loads the tokenizer from the vLLM-mounted model path, reconstructs the
full Supervisor prompt (schema + enum rules + algorithm + few-shots +
off-topic + user query), and reports:

  * prompt_tokens                          (what vLLM counts as input)
  * prompt_tokens + max_completion (512)   (total request size)
  * Verdict vs max-model-len=3072

Usage:
    docker compose run --rm --no-deps app python -m eval.count_supervisor_tokens
"""

from __future__ import annotations

import logging

from transformers import AutoTokenizer

from hse_prom_prog.agents.supervisor import SupervisorAgent
from hse_prom_prog.config import settings
from hse_prom_prog.llm.client import LLMClient

logger = logging.getLogger(__name__)

_MODEL_PATH = "/models/avibe-gptq-8bit"
_MAX_COMPLETION = 256  # matches supervisor.py max_tokens
_MAX_MODEL_LEN = 3072

_SAMPLE_QUERIES = [
    "Расскажи анекдот про Scrum-мастера",  # off_topic — worst case, hits all prompt branches
    "Что такое Scope Drop?",  # rag boundary
    "Какой scope drop у команды cthulhu?",  # sql metric
    "Покажи все задачи команды cthulhu и дай совет как улучшить",  # hybrid
]


def _build_supervisor_prompt(query: str) -> str:
    """Reconstruct the slow-path prompt by monkey-patching the LLM call."""
    captured: dict[str, str] = {}

    class _CaptureClient(LLMClient):
        def __init__(self) -> None:
            pass

        def invoke(  # type: ignore[override]
            self,
            prompt: str,
            response_format: object | None = None,
            max_tokens: int | None = None,
        ) -> str:
            captured["prompt"] = prompt
            # Return a minimal valid JSON so the pipeline doesn't crash
            return '{"intent": "general", "query_type": "simple", "entities": {}}'

    sup = SupervisorAgent(llm_client=_CaptureClient(), db_engine=None)
    # Force slow path by choosing a query without an issue key
    sup.process(query)
    return captured["prompt"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    logger.info("Loading tokenizer: %s", _MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH, trust_remote_code=True)

    budget = _MAX_MODEL_LEN - _MAX_COMPLETION
    logger.info(
        "Budget: max-model-len=%d, max_completion=%d → prompt must be < %d tokens",
        _MAX_MODEL_LEN,
        _MAX_COMPLETION,
        budget,
    )
    logger.info("Configured vllm temperature=%s", settings.vllm_temperature)

    any_fail = False
    for query in _SAMPLE_QUERIES:
        prompt = _build_supervisor_prompt(query)
        n_tokens = len(tokenizer.encode(prompt))
        total = n_tokens + _MAX_COMPLETION
        fits = total <= _MAX_MODEL_LEN
        mark = "OK " if fits else "FAIL"
        print(
            f"{mark}  prompt_tokens={n_tokens:5d}  "
            f"+completion={_MAX_COMPLETION}  total={total:5d}  "
            f"(limit={_MAX_MODEL_LEN})  | {query[:60]}"
        )
        if not fits:
            any_fail = True

    print()
    if any_fail:
        print(
            "Verdict: NO-GO — at least one prompt exceeds max-model-len.\n"
            "Options:\n"
            "  a) Compress more of the prompt (schema / few-shot / algorithm).\n"
            "  b) Lower max_completion in LLMClient.invoke() from 512 to 256.\n"
            "  c) Raise vllm max-model-len to 4096 (rebalance GPU budget):\n"
            "       vllm      gpu-memory-utilization=0.55\n"
            "       vllm-sql  gpu-memory-utilization=0.38 or max-model-len=4096"
        )
    else:
        print("Verdict: GO — Supervisor prompt fits the configured budget.")


if __name__ == "__main__":
    main()
