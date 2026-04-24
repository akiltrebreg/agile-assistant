"""Multi-turn Supervisor evaluation runner.

Exercises multi-turn cases from ``supervisor_golden_dataset.json``
(category ``multi_turn``) by replaying turns sequentially through the
Supervisor while injecting a synthetic ``ConversationContext`` built
from the ACTUAL output of prior turns — so upstream extraction errors
propagate downstream, the same way they would at runtime.

Metrics:
    * Entity carry-forward accuracy — on turns where an entity must
      carry from the prior turn (value in prev ``expected.entities``,
      same value in current ``expected.entities``, and the value is NOT
      mentioned verbatim in the current query), did the supervisor
      preserve it?
    * False carry-forward rate — on turns where the prior turn's value
      is NOT in current ``expected.entities``, did the supervisor wrongly
      inherit it?
    * Routing accuracy — ``query_type`` match on every turn.

Deploy gate: ≥ 85% carry-forward accuracy, 0% false carry-forward.

Regression reminder (Part 9.3 of the memory-layer plan): the four
existing eval pipelines (Supervisor 81, SQL 46, RAG 41, Response 40)
already invoke their agents with ``conversation_context=None`` and
``user_profile=None`` — that's the pre-memory code path, so a re-run
should produce baseline-equivalent numbers (≤ 1% drift is the gate).

Usage:
    docker compose run --rm --no-deps app \\
        python -m eval.run_multiturn_eval --experiment multiturn_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tabulate import tabulate

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
_GOLDEN = _EVAL_DIR / "supervisor_golden_dataset.json"
_RESULTS_DIR = _EVAL_DIR / "results"

# Fields eligible for carry-forward — mirrors entity_sanitizer._CARRY_FORWARD_FIELDS.
# Enums (issue_type/status/metric_name) and issue_key are intentionally excluded:
# the sanitizer only carries free-text context fields.
_CARRY_FORWARD_FIELDS: tuple[str, ...] = (
    "team_name",
    "sprint_name",
    "cluster",
    "assignee",
)

# Carry-forward accuracy floor and false-carry ceiling (deploy gate).
_CARRY_FORWARD_THRESHOLD = 0.85
_FALSE_CARRY_THRESHOLD = 0.0


# ── Comparators (shared shape with run_supervisor_eval) ──────


def _norm(v: Any) -> str:
    return str(v).strip().lower() if v is not None else ""


def _scalar_match(expected_val: Any, actual_val: Any) -> bool:
    e = _norm(expected_val)
    a = _norm(actual_val)
    if not e and not a:
        return True
    if not e or not a:
        return False
    return e == a or e in a or a in e


def _entity_value_match(expected_val: Any, actual_val: Any) -> bool:
    """Soft match with list support (team_name can be a list)."""
    if not isinstance(expected_val, list) and not isinstance(actual_val, list):
        return _scalar_match(expected_val, actual_val)
    exp_list = expected_val if isinstance(expected_val, list) else [expected_val]
    act_list = actual_val if isinstance(actual_val, list) else [actual_val]
    if not exp_list and not act_list:
        return True
    if not exp_list or not act_list:
        return False
    return all(any(_scalar_match(e, a) for a in act_list) for e in exp_list)


def _compare_entities(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    all_fields = set(expected) | set(actual)
    for field in all_fields:
        exp_val = expected.get(field)
        act_val = actual.get(field)
        exp_empty = exp_val in (None, "", [], {})
        act_empty = act_val in (None, "", [], {})
        if exp_empty and act_empty:
            continue
        if exp_empty and not act_empty:
            mismatches.append(f"{field}: +{act_val!r} (unexpected)")
            continue
        if not exp_empty and act_empty:
            mismatches.append(f"{field}: missing, want {exp_val!r}")
            continue
        if not _entity_value_match(exp_val, act_val):
            mismatches.append(f"{field}: {act_val!r} != {exp_val!r}")
    return (not mismatches), mismatches


# ── Carry-forward detection ──────────────────────────────────


def _mentioned_in_query(value: Any, query: str) -> bool:
    """Whether ``value`` appears verbatim (case-insensitive) in ``query``."""
    if value is None:
        return False
    ql = query.lower()
    if isinstance(value, list):
        return all(isinstance(v, str) and v.lower() in ql for v in value)
    return isinstance(value, str) and value.lower() in ql


def _classify_carry_forward(
    prev_expected_entities: dict[str, Any],
    curr_expected_entities: dict[str, Any],
    curr_query: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split carry-forward fields into MUST-carry and MUST-NOT-carry groups.

    Returns:
        (should_carry, should_not_carry) — each a ``{field: prev_value}`` map.
        * should_carry: field ∈ prev, same value in curr expected, value NOT
          uttered in the current query text.
        * should_not_carry: field ∈ prev, but curr expected either omits it
          or overrides with a different value.
    """
    should_carry: dict[str, Any] = {}
    should_not_carry: dict[str, Any] = {}
    for field in _CARRY_FORWARD_FIELDS:
        prev_val = prev_expected_entities.get(field)
        if prev_val in (None, "", [], {}):
            continue
        curr_val = curr_expected_entities.get(field)
        if curr_val in (None, "", [], {}):
            should_not_carry[field] = prev_val
            continue
        if _entity_value_match(prev_val, curr_val):
            # Only counts as "must carry" when the user did NOT repeat the
            # value — otherwise the LLM extracted it directly and carry-
            # forward logic wasn't exercised.
            if not _mentioned_in_query(curr_val, curr_query):
                should_carry[field] = prev_val
        else:
            should_not_carry[field] = prev_val
    return should_carry, should_not_carry


# ── Pipeline ─────────────────────────────────────────────────


def _build_supervisor() -> Any:
    from hse_prom_prog.agents.supervisor import SupervisorAgent
    from hse_prom_prog.database.connection import get_database
    from hse_prom_prog.llm.client import LLMClient

    client = LLMClient()
    try:
        db = get_database()
        engine = db.engine
        logger.info("[Multiturn Eval] DB engine ready — DB validation enabled")
    except Exception as e:
        engine = None
        logger.warning("[Multiturn Eval] DB unavailable (%s) — synonym-only mode", e)
    return SupervisorAgent(llm_client=client, db_engine=engine)


def _make_context(recent_turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Build a ``ConversationContext``-shaped dict from accumulated turns."""
    if not recent_turns:
        return None
    return {
        "summary": "",
        "recent_turns": list(recent_turns),
        "history_token_count": 0,
        "needs_summarization": False,
    }


def _append_turn_pair(
    recent_turns: list[dict[str, Any]],
    user_query: str,
    user_entities: dict[str, Any],
    turn_index: int,
) -> None:
    """Append the (user, assistant) pair to the rolling history buffer.

    The assistant turn is a neutral stub — its content doesn't carry
    domain data, so it can't bias classification on later turns.
    """
    recent_turns.append(
        {
            "role": "user",
            "content": user_query,
            "metadata": {"entities": dict(user_entities)},
            "turn_index": turn_index,
        }
    )
    recent_turns.append(
        {
            "role": "assistant",
            "content": "(вывод данных)",
            "metadata": {},
            "turn_index": turn_index + 1,
        }
    )


def _evaluate_turn(  # noqa: PLR0913
    *,
    turn_idx: int,
    turn: dict[str, Any],
    actual: dict[str, Any],
    prev_expected: dict[str, Any] | None,
    latency: float,
    error: str | None,
) -> dict[str, Any]:
    """Score one turn and return its result record."""
    expected = turn["expected"]
    expected_entities = expected.get("entities", {}) or {}
    actual_entities = actual.get("entities", {}) or {}
    query = turn["query"]

    routing_ok = actual.get("query_type") == expected["query_type"]
    intent_ok = actual.get("intent") == expected["intent"]
    entities_ok, entity_mismatches = _compare_entities(expected_entities, actual_entities)

    should_carry: dict[str, Any] = {}
    should_not_carry: dict[str, Any] = {}
    carry_hits: list[str] = []
    carry_misses: list[str] = []
    false_carries: list[str] = []
    if prev_expected is not None:
        should_carry, should_not_carry = _classify_carry_forward(
            prev_expected, expected_entities, query
        )
        for field, want in should_carry.items():
            if _entity_value_match(want, actual_entities.get(field)):
                carry_hits.append(field)
            else:
                carry_misses.append(field)
        for field, prev_val in should_not_carry.items():
            act_val = actual_entities.get(field)
            if act_val in (None, "", [], {}):
                continue
            # False carry only if the actual value MATCHES the prev value
            # (i.e. it's the stale one), not just any wrong value.
            if _entity_value_match(prev_val, act_val) and not _mentioned_in_query(prev_val, query):
                false_carries.append(field)

    return {
        "turn_index": turn_idx,
        "query": query,
        "expected": expected,
        "actual": {
            "intent": actual.get("intent"),
            "query_type": actual.get("query_type"),
            "entities": actual_entities,
        },
        "routing_match": routing_ok,
        "intent_match": intent_ok,
        "entities_match": entities_ok,
        "entity_mismatches": entity_mismatches,
        "carry_forward": {
            "should_carry": {k: v for k, v in should_carry.items()},
            "should_not_carry": {k: v for k, v in should_not_carry.items()},
            "hits": carry_hits,
            "misses": carry_misses,
            "false_carries": false_carries,
        },
        "latency_s": round(latency, 3),
        "error": error,
    }


def _run_case(supervisor: Any, case: dict[str, Any]) -> dict[str, Any]:
    turns = case["turns"]
    recent_turns: list[dict[str, Any]] = []
    turn_results: list[dict[str, Any]] = []
    prev_expected_entities: dict[str, Any] | None = None

    for idx, turn in enumerate(turns):
        query = turn["query"]
        ctx = _make_context(recent_turns)

        t0 = time.perf_counter()
        try:
            actual = supervisor.process(query, conversation_context=ctx)
            err = None
        except Exception as e:
            actual = {"intent": "general", "entities": {}, "query_type": "simple"}
            err = str(e)
            logger.error("[Multiturn Eval] Error on case #%d turn %d: %s", case["id"], idx, e)
        latency = time.perf_counter() - t0

        record = _evaluate_turn(
            turn_idx=idx,
            turn=turn,
            actual=actual,
            prev_expected=prev_expected_entities,
            latency=latency,
            error=err,
        )
        turn_results.append(record)

        status_bits = []
        if not record["routing_match"]:
            status_bits.append(f"qt={actual.get('query_type')}→{turn['expected']['query_type']}")
        if record["carry_forward"]["misses"]:
            status_bits.append(f"miss={record['carry_forward']['misses']}")
        if record["carry_forward"]["false_carries"]:
            status_bits.append(f"false={record['carry_forward']['false_carries']}")
        status = "PASS" if not status_bits else "FAIL"
        logger.info(
            "  case #%d turn %d [%s]: %s (%.2fs)",
            case["id"],
            idx,
            status,
            "; ".join(status_bits) or "all match",
            latency,
        )

        _append_turn_pair(
            recent_turns,
            query,
            actual.get("entities", {}) or {},
            turn_index=idx * 2,
        )
        prev_expected_entities = turn["expected"].get("entities", {}) or {}

    return {
        "id": case["id"],
        "subcategory": case.get("subcategory", "unspecified"),
        "notes": case.get("notes", ""),
        "turns": turn_results,
    }


def _run_eval(supervisor: Any, dataset: list[dict]) -> list[dict]:
    results = []
    for i, case in enumerate(dataset, 1):
        logger.info(
            "[%d/%d] case #%d [%s] (%d turns)",
            i,
            len(dataset),
            case["id"],
            case.get("subcategory", "?"),
            len(case["turns"]),
        )
        results.append(_run_case(supervisor, case))
    return results


# ── Aggregation ──────────────────────────────────────────────


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_turns = 0
    routing_ok = 0
    entities_ok = 0
    carry_opportunities = 0
    carry_hits = 0
    false_carry_opportunities = 0
    false_carry_events = 0

    per_sub: dict[str, dict[str, int]] = {}

    for case in results:
        sub = case["subcategory"]
        bucket = per_sub.setdefault(
            sub,
            {
                "cases": 0,
                "turns": 0,
                "routing_ok": 0,
                "carry_opportunities": 0,
                "carry_hits": 0,
                "false_opportunities": 0,
                "false_events": 0,
            },
        )
        bucket["cases"] += 1
        for turn in case["turns"]:
            total_turns += 1
            bucket["turns"] += 1
            if turn["routing_match"]:
                routing_ok += 1
                bucket["routing_ok"] += 1
            if turn["entities_match"]:
                entities_ok += 1
            cf = turn["carry_forward"]
            n_carry = len(cf["should_carry"])
            carry_opportunities += n_carry
            bucket["carry_opportunities"] += n_carry
            hits = len(cf["hits"])
            carry_hits += hits
            bucket["carry_hits"] += hits
            n_false = len(cf["should_not_carry"])
            false_carry_opportunities += n_false
            bucket["false_opportunities"] += n_false
            false_events = len(cf["false_carries"])
            false_carry_events += false_events
            bucket["false_events"] += false_events

    def _pct(n: int, d: int) -> float:
        return round(n / d, 3) if d else 0.0

    return {
        "cases": len(results),
        "turns": total_turns,
        "routing_accuracy": _pct(routing_ok, total_turns),
        "entities_accuracy": _pct(entities_ok, total_turns),
        "carry_forward_opportunities": carry_opportunities,
        "carry_forward_hits": carry_hits,
        "carry_forward_accuracy": _pct(carry_hits, carry_opportunities),
        "false_carry_opportunities": false_carry_opportunities,
        "false_carry_events": false_carry_events,
        "false_carry_rate": _pct(false_carry_events, false_carry_opportunities),
        "per_subcategory": per_sub,
    }


# ── Reporting ────────────────────────────────────────────────


def _print_summary(results: list[dict[str, Any]], agg: dict[str, Any], experiment: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Multi-turn Supervisor Eval: {experiment}")
    print(f"Cases: {agg['cases']}, turns: {agg['turns']}")
    print(f"{'=' * 60}\n")

    print(
        f"  Routing accuracy         : {agg['routing_accuracy'] * 100:.1f}%  "
        f"({int(agg['routing_accuracy'] * agg['turns'])}/{agg['turns']})"
    )
    print(
        f"  Entity carry-forward acc : {agg['carry_forward_accuracy'] * 100:.1f}%  "
        f"({agg['carry_forward_hits']}/{agg['carry_forward_opportunities']})"
    )
    print(
        f"  False carry-forward rate : {agg['false_carry_rate'] * 100:.1f}%  "
        f"({agg['false_carry_events']}/{agg['false_carry_opportunities']})"
    )

    rows = []
    for sub, b in sorted(agg["per_subcategory"].items()):
        rows.append(
            [
                sub,
                b["cases"],
                b["turns"],
                f"{b['routing_ok']}/{b['turns']}",
                f"{b['carry_hits']}/{b['carry_opportunities']}"
                if b["carry_opportunities"]
                else "—",
                f"{b['false_events']}/{b['false_opportunities']}"
                if b["false_opportunities"]
                else "—",
            ]
        )
    print("\nPer-subcategory:")
    print(
        tabulate(
            rows,
            headers=["Subcategory", "Cases", "Turns", "Routing", "Carry", "False carry"],
            tablefmt="simple",
        )
    )

    # Failure detail — show turns that failed routing or carry-forward
    failures: list[tuple[int, int, dict]] = []
    for case in results:
        for turn in case["turns"]:
            cf = turn["carry_forward"]
            if turn["routing_match"] and not cf["misses"] and not cf["false_carries"]:
                continue
            failures.append((case["id"], turn["turn_index"], turn))
    if failures:
        print(f"\nFailed turns ({len(failures)}):")
        for case_id, turn_idx, t in failures:
            q = t["query"] if len(t["query"]) <= 50 else t["query"][:47] + "..."
            tags = []
            if not t["routing_match"]:
                tags.append("ROUTING")
            if t["carry_forward"]["misses"]:
                tags.append(f"MISS:{t['carry_forward']['misses']}")
            if t["carry_forward"]["false_carries"]:
                tags.append(f"FALSE:{t['carry_forward']['false_carries']}")
            print(f"  #{case_id}.t{turn_idx} [{','.join(tags)}] {q}")
            exp = t["expected"]
            act = t["actual"]
            print(
                f"    expected: qt={exp['query_type']}, intent={exp['intent']}, "
                f"entities={exp.get('entities', {})}"
            )
            print(
                f"    actual  : qt={act['query_type']}, intent={act['intent']}, "
                f"entities={act['entities']}"
            )
    print()


# ── Deploy gate ──────────────────────────────────────────────


def _check_deploy_gate(agg: dict[str, Any]) -> bool:
    carry_ok = (
        agg["carry_forward_opportunities"] == 0
        or agg["carry_forward_accuracy"] >= _CARRY_FORWARD_THRESHOLD
    )
    false_ok = agg["false_carry_rate"] <= _FALSE_CARRY_THRESHOLD

    rows = [
        (
            "Carry-forward accuracy",
            f"{agg['carry_forward_accuracy'] * 100:.1f}%",
            f"≥ {_CARRY_FORWARD_THRESHOLD * 100:.0f}%",
            carry_ok,
        ),
        (
            "False carry-forward rate",
            f"{agg['false_carry_rate'] * 100:.1f}%",
            f"≤ {_FALSE_CARRY_THRESHOLD * 100:.0f}%",
            false_ok,
        ),
    ]

    print("\n" + "=" * 60)
    print("DEPLOY GATE")
    print("=" * 60)
    print(
        tabulate(
            [[name, actual, thr, "PASS" if ok else "FAIL"] for name, actual, thr, ok in rows],
            headers=["Criterion", "Result", "Threshold", "Verdict"],
            tablefmt="simple",
        )
    )
    all_pass = carry_ok and false_ok
    if all_pass:
        print("\nVerdict: GO — safe to deploy.\n")
    else:
        failed = [name for name, _, _, ok in rows if not ok]
        print(f"\nVerdict: NO-GO — failed: {failed}")
        if "False carry-forward rate" in failed:
            print(
                "  → Supervisor inherits stale entities across turns. "
                "Check entity_sanitizer carry-forward guard and _ANAPHORA_MARKERS."
            )
        if "Carry-forward accuracy" in failed:
            print(
                "  → Supervisor drops legitimate context. "
                "Check prev_entities plumbing + sanitizer layer 6."
            )
        print()

    return all_pass


# ── Main ─────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run multi-turn Supervisor evaluation")
    parser.add_argument("--experiment", default="multiturn_baseline")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    with _GOLDEN.open() as f:
        full_dataset = json.load(f)
    dataset = [c for c in full_dataset if c.get("category") == "multi_turn"]
    logger.info("Loaded %d multi-turn cases from %s", len(dataset), _GOLDEN.name)
    if not dataset:
        logger.error("No multi-turn cases found — nothing to run.")
        return

    supervisor = _build_supervisor()
    results = _run_eval(supervisor, dataset)
    agg = _aggregate(results)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": args.experiment,
        "timestamp": timestamp,
        "aggregate": agg,
        "results": results,
    }

    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = _RESULTS_DIR / f"{args.experiment}_{timestamp}.json"
    with out_path.open("w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Saved results to %s", out_path)

    _print_summary(results, agg, args.experiment)
    _check_deploy_gate(agg)


if __name__ == "__main__":
    main()
