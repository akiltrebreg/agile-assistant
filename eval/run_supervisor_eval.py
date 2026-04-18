"""Supervisor Agent evaluation runner.

Classifies each query in supervisor_golden_dataset.json through
SupervisorAgent and reports routing accuracy, entity extraction quality,
confusion matrix over query_type, and fast-path rate.

Usage:
    docker compose run --rm --no-deps app \\
        python -m eval.run_supervisor_eval --experiment supervisor_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tabulate import tabulate

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
_GOLDEN = _EVAL_DIR / "supervisor_golden_dataset.json"
_RESULTS_DIR = _EVAL_DIR / "results"

# Which entity fields we compare (others are noise / optional)
_KEY_ENTITY_FIELDS = ("issue_key", "team_name", "sprint_name", "metric_name")
_SOFT_ENTITY_FIELDS = ("issue_type", "status", "assignee", "cluster")


# ── Comparators ──────────────────────────────────────────────


def _norm(v: Any) -> str:
    return str(v).strip().lower() if v is not None else ""


def _scalar_match(expected_val: Any, actual_val: Any) -> bool:
    """Match two scalars: case-insensitive, substring either way."""
    e = _norm(expected_val)
    a = _norm(actual_val)
    if not e and not a:
        return True
    if not e or not a:
        return False
    return e == a or e in a or a in e


def _entity_match(expected_val: Any, actual_val: Any) -> bool:
    """Soft match: case-insensitive, substring either way.

    Supports lists on either side (for team_name when multiple teams
    are mentioned). A list matches if ALL expected items appear in
    actual (list or scalar) via substring logic.
    """
    if not isinstance(expected_val, list) and not isinstance(actual_val, list):
        return _scalar_match(expected_val, actual_val)

    exp_list = expected_val if isinstance(expected_val, list) else [expected_val]
    act_list = actual_val if isinstance(actual_val, list) else [actual_val]
    if not exp_list and not act_list:
        return True
    if not exp_list or not act_list:
        return False
    return all(any(_scalar_match(e, a) for a in act_list) for e in exp_list)


def _compare_entities(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check that every expected key entity is present and matches (soft).

    Extra actual fields are ignored. Missing expected fields count as failure.
    """
    mismatches: list[str] = []
    for field in _KEY_ENTITY_FIELDS + _SOFT_ENTITY_FIELDS:
        if field not in expected:
            continue
        exp_val = expected[field]
        act_val = actual.get(field)
        if not _entity_match(exp_val, act_val):
            mismatches.append(f"{field}: {act_val!r} != {exp_val!r}")
    return (not mismatches), mismatches


# ── Pipeline ─────────────────────────────────────────────────


def _build_supervisor() -> Any:
    """Build Supervisor with main LLM client."""
    from hse_prom_prog.agents.supervisor import SupervisorAgent
    from hse_prom_prog.llm.client import LLMClient

    client = LLMClient()
    return SupervisorAgent(llm_client=client)


def _run_eval(supervisor: Any, dataset: list[dict]) -> list[dict]:
    results = []
    total = len(dataset)
    for i, case in enumerate(dataset, 1):
        cid = case["id"]
        query = case["query"]
        expected = case["expected"]
        category = case["category"]

        logger.info("[%d/%d] #%d [%s] %s", i, total, cid, category, query[:60])

        t0 = time.perf_counter()
        try:
            output = supervisor.process(query)
            err = None
        except Exception as e:
            output = {"intent": "general", "entities": {}, "query_type": "simple"}
            err = str(e)
            logger.error("[Supervisor Eval] Error on #%d: %s", cid, e)
        latency = time.perf_counter() - t0

        # Detect fast-path: if query has issue_key pattern, Supervisor hits fast-path
        fast_path_fired = bool(re.search(r"\b[A-Z]{2,}-\d+\b", query)) and (
            "issue_key" in output.get("entities", {})
        )

        # Three levels of match
        act_qt = output.get("query_type")
        act_intent = output.get("intent")
        act_entities = output.get("entities", {})
        routing_ok = act_qt == expected["query_type"]
        intent_ok = act_intent == expected["intent"]
        entities_ok, entity_mismatches = _compare_entities(
            expected.get("entities", {}), act_entities
        )

        exact = routing_ok and intent_ok and entities_ok
        partial = routing_ok and intent_ok

        results.append(
            {
                "id": cid,
                "query": query,
                "category": category,
                "expected": expected,
                "actual": {
                    "intent": act_intent,
                    "query_type": act_qt,
                    "entities": act_entities,
                },
                "routing_match": routing_ok,
                "intent_match": intent_ok,
                "entities_match": entities_ok,
                "partial_match": partial,
                "exact_match": exact,
                "entity_mismatches": entity_mismatches,
                "fast_path": fast_path_fired,
                "latency_s": round(latency, 3),
                "error": err,
            }
        )

        status = "PASS" if exact else ("PART" if partial else "FAIL")
        detail_bits = []
        if not routing_ok:
            detail_bits.append(f"qt: {act_qt}→{expected['query_type']}")
        if not intent_ok:
            detail_bits.append(f"intent: {act_intent}→{expected['intent']}")
        if not entities_ok:
            detail_bits.append("entities: " + "; ".join(entity_mismatches))
        detail = "; ".join(detail_bits) if detail_bits else "all match"
        logger.info("  %s: %s (%.2fs)", status, detail[:100], latency)

    return results


# ── Reporting ────────────────────────────────────────────────


def _confusion_matrix(results: list[dict]) -> str:
    """Build query_type confusion matrix (rows=expected, cols=predicted)."""
    labels = ["sql", "rag", "hybrid", "simple"]
    matrix: dict[str, dict[str, int]] = {e: {p: 0 for p in labels} for e in labels}
    for r in results:
        exp = r["expected"]["query_type"]
        act = r["actual"]["query_type"] or "simple"
        if exp in matrix and act in matrix[exp]:
            matrix[exp][act] += 1
    rows = [[e] + [matrix[e][p] for p in labels] for e in labels]
    return tabulate(
        rows,
        headers=["expected \\ actual", *labels],
        tablefmt="simple",
    )


def _print_summary(results: list[dict], experiment: str) -> None:
    total = len(results)
    exact = sum(1 for r in results if r["exact_match"])
    partial = sum(1 for r in results if r["partial_match"])
    routing = sum(1 for r in results if r["routing_match"])

    print(f"\n{'=' * 60}")
    print(f"Supervisor Eval: {experiment}  ({total} cases)")
    print(f"{'=' * 60}\n")
    print(f"  Routing accuracy : {routing}/{total} ({routing / total * 100:.1f}%)")
    print(f"  Partial match    : {partial}/{total} ({partial / total * 100:.1f}%)")
    print(f"  Exact match      : {exact}/{total} ({exact / total * 100:.1f}%)")

    # Per-category
    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)
    print("\nPer-category:")
    cat_rows = []
    for cat, items in sorted(categories.items()):
        n = len(items)
        r_ok = sum(1 for x in items if x["routing_match"])
        e_ok = sum(1 for x in items if x["exact_match"])
        cat_rows.append(
            [
                cat,
                n,
                f"{r_ok}/{n}",
                f"{r_ok / n * 100:.0f}%",
                f"{e_ok}/{n}",
                f"{e_ok / n * 100:.0f}%",
            ]
        )
    print(
        tabulate(
            cat_rows,
            headers=["Category", "N", "Routing", "R%", "Exact", "E%"],
            tablefmt="simple",
        )
    )

    # Confusion matrix
    print("\nConfusion matrix (query_type, expected rows × actual cols):")
    print(_confusion_matrix(results))

    # Fast-path
    fp_cases = [r for r in results if "-" in r["query"] and r["query"]]
    fp_fired = sum(1 for r in fp_cases if r["fast_path"])
    fp_eligible = sum(1 for r in fp_cases if r["expected"]["entities"].get("issue_key"))
    if fp_eligible:
        print(
            f"\nFast-path: fired on {fp_fired}/{fp_eligible} "
            f"eligible queries ({fp_fired / fp_eligible * 100:.0f}%)"
        )

    # Latency
    slow_path = [r["latency_s"] for r in results if not r["fast_path"] and not r["error"]]
    fast_path = [r["latency_s"] for r in results if r["fast_path"]]
    if slow_path:
        s = sorted(slow_path)
        p50 = s[len(s) // 2]
        p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
        print(
            f"\nLatency slow-path: n={len(slow_path)}, "
            f"p50={p50:.2f}s, p95={p95:.2f}s, max={max(slow_path):.2f}s"
        )
    if fast_path:
        print(
            f"Latency fast-path: n={len(fast_path)}, "
            f"avg={sum(fast_path) / len(fast_path) * 1000:.1f}ms"
        )

    # Failures
    failures = [r for r in results if not r["exact_match"]]
    if failures:
        print(f"\nFailed ({len(failures)}):")
        for r in failures:
            q = r["query"] if len(r["query"]) <= 50 else r["query"][:47] + "..."
            tag = (
                "ROUTING"
                if not r["routing_match"]
                else ("INTENT" if not r["intent_match"] else "ENTITIES")
            )
            print(f"  #{r['id']} [{r['category']}] [{tag}] {q}")
            exp, act = r["expected"], r["actual"]
            print(
                f"    expected: qt={exp['query_type']}, intent={exp['intent']}, "
                f"entities={exp.get('entities', {})}"
            )
            print(
                f"    actual  : qt={act['query_type']}, intent={act['intent']}, "
                f"entities={act['entities']}"
            )
            if r["entity_mismatches"]:
                print(f"    mismatches: {'; '.join(r['entity_mismatches'])}")
    print()


# ── Main ─────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Supervisor Agent evaluation")
    parser.add_argument("--experiment", default="supervisor_baseline")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    with _GOLDEN.open() as f:
        dataset = json.load(f)
    logger.info("Loaded %d Supervisor test cases", len(dataset))

    supervisor = _build_supervisor()

    results = _run_eval(supervisor, dataset)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    total = len(results)
    exact = sum(1 for r in results if r["exact_match"])
    routing = sum(1 for r in results if r["routing_match"])

    output = {
        "experiment": args.experiment,
        "timestamp": timestamp,
        "aggregate": {
            "total": total,
            "routing_match": routing,
            "exact_match": exact,
            "partial_match": sum(1 for r in results if r["partial_match"]),
            "routing_accuracy": round(routing / total, 3) if total else 0,
            "exact_accuracy": round(exact / total, 3) if total else 0,
        },
        "results": results,
    }

    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = _RESULTS_DIR / f"{args.experiment}_{timestamp}.json"
    with out_path.open("w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Saved results to %s", out_path)

    _print_summary(results, args.experiment)


if __name__ == "__main__":
    main()
