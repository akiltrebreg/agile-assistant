#!/usr/bin/env bash
# Memory Layer end-to-end verification.
#
# Runs four levels of checks against the running stack and reports
# PASS/FAIL per check + an aggregate verdict at the end. Each level can
# be skipped independently. The script never aborts on the first failure
# — it always finishes the run so you see the full picture.
#
# Usage:
#   ./scripts/verify_memory.sh [--skip-unit] [--skip-eval] [--skip-e2e]
#                              [--regression] [-h|--help]
#
# Levels:
#   1. Unit       — pytest tests/test_memory.py + tests/test_api_memory.py
#                   (mocks only, no live stack).
#   2. Eval       — eval/run_multiturn_eval.py against live vllm.
#                   Requires: postgres + vllm healthy.
#   3. E2E        — POST /tasks via api, then DB inspection.
#                   Requires: postgres + vllm + vllm-sql + qdrant +
#                             redis + api + celery-worker.
#   4. Regression — 4 baseline evals (Supervisor/SQL/RAG/Response).
#                   Opt-in with --regression. Requires full stack.
#                   ~10 min on RTX 4090.
#
# Exit codes:
#   0 — all enabled checks passed
#   1 — one or more checks failed
#   2 — prerequisite missing (Docker / .env)

set -uo pipefail
IFS=$'\n\t'

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || { echo "cannot cd to repo root: $ROOT" >&2; exit 2; }

# Unique prefix for test users — lets cleanup find rows we created.
VERIFY_UID="verify-$(date +%s)-$$"
RESULTS_DIR="$ROOT/eval/results"
LOG_FILE="$ROOT/.verify_memory_$(date +%Y%m%d_%H%M%S).log"

RUN_UNIT=1
RUN_EVAL=1
RUN_E2E=1
RUN_REGRESSION=0

POLL_DEADLINE_SEC=120   # per-task timeout when polling /tasks/{id}
PROFILE_REFRESH_SEC=8   # wait for update_profile_async to settle

# ─────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────

if [[ -t 1 ]]; then
    RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'
    CYN=$'\033[36m'; DIM=$'\033[2m'; CLR=$'\033[0m'
else
    RED=""; GRN=""; YLW=""; CYN=""; DIM=""; CLR=""
fi

PASS_COUNT=0
FAIL_COUNT=0
FAILED_CHECKS=()

note() { echo "${DIM}▸${CLR} $*"; }
pass() {
    echo "${GRN}✓ PASS${CLR}: $*"
    PASS_COUNT=$((PASS_COUNT + 1))
}
fail() {
    echo "${RED}✗ FAIL${CLR}: $*"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAILED_CHECKS+=("$*")
}
skip() {
    echo "${YLW}- SKIP${CLR}: $*"
}
section() {
    echo
    echo "${CYN}════════════════════════════════════════════════════════════${CLR}"
    echo "${CYN}$*${CLR}"
    echo "${CYN}════════════════════════════════════════════════════════════${CLR}"
}

usage() {
    # Prints the leading comment block (lines 2 until the first non-comment).
    awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"
    exit 0
}

# ─────────────────────────────────────────────────────────────
# Arg parsing
# ─────────────────────────────────────────────────────────────

for arg in "$@"; do
    case "$arg" in
        --skip-unit)     RUN_UNIT=0 ;;
        --skip-eval)     RUN_EVAL=0 ;;
        --skip-e2e)      RUN_E2E=0 ;;
        --regression)    RUN_REGRESSION=1 ;;
        -h|--help)       usage ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# ─────────────────────────────────────────────────────────────
# Prerequisite checks
# ─────────────────────────────────────────────────────────────

section "Prerequisites"

if ! command -v docker >/dev/null 2>&1; then
    fail "docker not on PATH"; exit 2
fi
pass "docker present"

if ! docker compose version >/dev/null 2>&1; then
    fail "docker compose v2 not available"; exit 2
fi
pass "docker compose v2 present"

if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 not on PATH (used for JSON parsing)"; exit 2
fi
pass "python3 present"

if [[ ! -f .env ]]; then
    fail ".env missing — cp .env.example .env first"; exit 2
fi
pass ".env present"

# ─────────────────────────────────────────────────────────────
# Service health probes — direct, never trust `docker compose ps`
# wording differences across versions.
# ─────────────────────────────────────────────────────────────

probe_postgres() {
    docker compose exec -T postgres \
        pg_isready -U hse_user -d hse_jira_db >/dev/null 2>&1
}
probe_vllm() {
    docker compose exec -T vllm \
        sh -c 'curl -sf http://localhost:8000/v1/models -o /dev/null' >/dev/null 2>&1
}
probe_vllm_sql() {
    docker compose exec -T vllm-sql \
        sh -c 'curl -sf http://localhost:8000/v1/models -o /dev/null' >/dev/null 2>&1
}
probe_qdrant() {
    docker compose exec -T qdrant \
        sh -c 'curl -sf http://localhost:6333/readyz -o /dev/null' >/dev/null 2>&1
}
probe_api() {
    docker compose exec -T api \
        sh -c 'curl -sf http://localhost:8080/health -o /dev/null' >/dev/null 2>&1
}
probe_celery() {
    # No HTTP endpoint — verify the worker process is running.
    docker compose exec -T celery-worker \
        sh -c 'pgrep -f "celery.*worker" >/dev/null' >/dev/null 2>&1
}

HAVE_POSTGRES=0
HAVE_VLLM=0
HAVE_VLLM_SQL=0
HAVE_QDRANT=0
HAVE_API=0
HAVE_CELERY=0

probe_postgres && HAVE_POSTGRES=1
probe_vllm     && HAVE_VLLM=1
probe_vllm_sql && HAVE_VLLM_SQL=1
probe_qdrant   && HAVE_QDRANT=1
probe_api      && HAVE_API=1
probe_celery   && HAVE_CELERY=1

note "service probes:"
echo "    postgres=$HAVE_POSTGRES vllm=$HAVE_VLLM vllm-sql=$HAVE_VLLM_SQL"
echo "    qdrant=$HAVE_QDRANT api=$HAVE_API celery-worker=$HAVE_CELERY"

# ─────────────────────────────────────────────────────────────
# Tool helpers
# ─────────────────────────────────────────────────────────────

# Run a single SQL statement. Output rows are tab-separated, one per line.
# `-t -A` strip headers and align padding.
psql_q() {
    docker compose exec -T postgres \
        psql -U hse_user -d hse_jira_db -t -A -F $'\t' -c "$1"
}

# JSON accessor. Reads from stdin, prints the result of evaluating the
# Python suffix on the parsed object, e.g. jget "['task_id']".
jget() {
    python3 -c "import json,sys; print(json.load(sys.stdin)$1)"
}

# POST /tasks {query, conversation_id?, user_id?} via the api container.
# Echoes the response JSON. Empty conv/uid args mean "do not send".
post_task() {
    local query="$1" cid="${2:-}" uid="${3:-}"
    local payload
    payload=$(python3 -c '
import json, sys
out = {"query": sys.argv[1]}
if sys.argv[2]:
    out["conversation_id"] = sys.argv[2]
if sys.argv[3]:
    out["user_id"] = sys.argv[3]
print(json.dumps(out, ensure_ascii=False))
' "$query" "$cid" "$uid")
    docker compose exec -T api \
        curl -sS -X POST -H 'Content-Type: application/json' \
        --data "$payload" http://localhost:8080/tasks
}

# Poll GET /tasks/{id} until COMPLETED or FAILED, or until timeout.
# Echoes the final task JSON. Returns 0 on terminal status, 1 on timeout.
poll_task() {
    local task_id="$1"
    local deadline=$(( $(date +%s) + POLL_DEADLINE_SEC ))
    local resp status
    resp=""
    while (( $(date +%s) < deadline )); do
        resp=$(docker compose exec -T api \
            curl -sS "http://localhost:8080/tasks/$task_id" 2>/dev/null || echo "")
        if [[ -n "$resp" ]]; then
            status=$(echo "$resp" | jget "['status']" 2>/dev/null || echo "")
            if [[ "$status" == "COMPLETED" || "$status" == "FAILED" ]]; then
                echo "$resp"
                return 0
            fi
        fi
        sleep 2
    done
    echo "$resp"
    return 1
}

# Fire `post_task` + `poll_task` in sequence, pushing their outputs through
# named tmp files (stdout collisions are awkward in bash). Echoes the final
# task JSON on stdout. Returns the poll exit code.
run_turn() {
    local query="$1" cid="${2:-}" uid="${3:-}"
    local create_resp task_id
    create_resp=$(post_task "$query" "$cid" "$uid")
    task_id=$(echo "$create_resp" | jget "['task_id']" 2>/dev/null || echo "")
    if [[ -z "$task_id" ]]; then
        echo "$create_resp" >&2
        return 2
    fi
    poll_task "$task_id"
}

# ─────────────────────────────────────────────────────────────
# LEVEL 1 — Unit tests
# ─────────────────────────────────────────────────────────────

run_unit_tests() {
    section "Level 1 — Unit tests (mocks only, no live stack)"
    if (( RUN_UNIT == 0 )); then skip "unit tests"; return; fi

    note "pytest tests/test_memory.py + tests/test_api_memory.py"
    if docker compose run --rm --no-deps \
        -v "$ROOT/tests:/app/tests" app \
        python -m pytest tests/test_memory.py tests/test_api_memory.py -q \
        >> "$LOG_FILE" 2>&1; then
        pass "unit tests (test_memory + test_api_memory)"
    else
        fail "unit tests — see $LOG_FILE"
    fi
}

# ─────────────────────────────────────────────────────────────
# LEVEL 2 — Multi-turn eval
# ─────────────────────────────────────────────────────────────

run_multiturn_eval() {
    section "Level 2 — Multi-turn eval"
    if (( RUN_EVAL == 0 )); then skip "multi-turn eval"; return; fi
    if (( HAVE_VLLM == 0 )); then
        skip "multi-turn eval — vllm not healthy"; return
    fi
    if (( HAVE_POSTGRES == 0 )); then
        note "postgres not healthy — sanitizer in synonym-only mode (still valid)"
    fi

    local exp_id
    exp_id="verify_$(date +%Y%m%d_%H%M%S)"
    note "running eval.run_multiturn_eval --experiment $exp_id"
    if ! docker compose run --rm --no-deps app \
        python -m eval.run_multiturn_eval --experiment "$exp_id" \
        >> "$LOG_FILE" 2>&1; then
        fail "multi-turn eval crashed — see $LOG_FILE"
        return
    fi

    # Find the result file (most recent that matches the experiment id).
    local result
    result=$(find "$RESULTS_DIR" -name "${exp_id}_*.json" -print 2>/dev/null \
             | sort -r | head -1)
    if [[ -z "$result" ]]; then
        fail "multi-turn eval — no result JSON written"; return
    fi
    note "result: $result"

    # Parse aggregate metrics.
    local cf_acc fc_rate
    cf_acc=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d['aggregate']['carry_forward_accuracy'])
" "$result" 2>/dev/null || echo "")
    fc_rate=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d['aggregate']['false_carry_rate'])
" "$result" 2>/dev/null || echo "")

    if [[ -z "$cf_acc" || -z "$fc_rate" ]]; then
        fail "could not parse aggregate metrics from $result"; return
    fi
    note "carry_forward_accuracy=$cf_acc, false_carry_rate=$fc_rate"

    if python3 -c "import sys; sys.exit(0 if float('$cf_acc') >= 0.85 else 1)"; then
        pass "carry-forward accuracy ≥ 85% (got $cf_acc)"
    else
        fail "carry-forward accuracy < 85% (got $cf_acc)"
    fi
    if python3 -c "import sys; sys.exit(0 if float('$fc_rate') == 0.0 else 1)"; then
        pass "false carry-forward rate = 0% (got $fc_rate)"
    else
        fail "false carry-forward rate > 0% (got $fc_rate)"
    fi
}

# ─────────────────────────────────────────────────────────────
# LEVEL 3 — End-to-end via API + DB
# ─────────────────────────────────────────────────────────────

# 3a. Short-term: anaphora carry-forward through one conversation.
run_e2e_short_term() {
    note "E2E 3a: short-term carry-forward (anaphora 'у них')"
    local uid="${VERIFY_UID}-st"

    local resp1 cid resp2
    resp1=$(run_turn "Velocity cthulhu?" "" "$uid") || {
        fail "E2E 3a — turn 1 timed out"; return
    }
    cid=$(echo "$resp1" | jget "['conversation_id']" 2>/dev/null || echo "")
    if [[ -z "$cid" || "$cid" == "None" ]]; then
        fail "E2E 3a — no conversation_id in turn 1 response"; return
    fi

    resp2=$(run_turn "А у них scope drop?" "$cid" "$uid") || {
        fail "E2E 3a — turn 2 timed out"; return
    }

    # Allow save_turn to commit before we read messages.
    sleep 2

    local team_in_t2
    team_in_t2=$(psql_q "
        SELECT metadata->'entities'->>'team_name'
        FROM messages
        WHERE conversation_id = '$cid'
          AND role = 'user'
          AND turn_index = 2;
    " | head -1)

    if [[ "$team_in_t2" == "cthulhu" ]]; then
        pass "E2E 3a: turn 2 metadata.entities.team_name='cthulhu' (carried over)"
    else
        fail "E2E 3a: turn 2 has team_name='$team_in_t2', expected 'cthulhu'"
    fi
}

# 3b. Profile gate: 3 messages must NOT yield default_team yet.
run_e2e_profile_gate() {
    note "E2E 3b: profile gate (3 messages → preferences stays empty)"
    local uid="${VERIFY_UID}-gate"
    local cid=""

    for q in "Velocity lpop" "Метрики lpop за Q1" "Задачи команды lpop"; do
        local resp
        resp=$(run_turn "$q" "$cid" "$uid") || {
            fail "E2E 3b — request '$q' timed out"; return
        }
        cid=$(echo "$resp" | jget "['conversation_id']" 2>/dev/null || echo "$cid")
    done

    # Wait for the last update_profile_async to settle.
    sleep "$PROFILE_REFRESH_SEC"

    local prefs
    prefs=$(psql_q "
        SELECT preferences::text
        FROM user_profiles
        WHERE external_id = '$uid';
    " | head -1)

    # Acceptable: no row at all, or empty JSON object.
    if [[ -z "$prefs" || "$prefs" == "{}" ]]; then
        pass "E2E 3b: preferences empty after 3 messages (gate active)"
    else
        fail "E2E 3b: preferences='$prefs', expected empty"
    fi
}

# 3c. Profile injection: 6 messages → default_team set, then a query without
# team triggers the supervisor's profile fallback.
run_e2e_profile_injection() {
    note "E2E 3c: default_team injection after 6 messages"
    local uid="${VERIFY_UID}-prof"
    local cid=""
    local queries=(
        "Velocity cthulhu"
        "Метрики команды cthulhu за Q1"
        "Задачи команды cthulhu"
        "Done total cthulhu"
        "Scope drop cthulhu"
        "Какие баги у cthulhu"
    )

    for q in "${queries[@]}"; do
        local resp
        resp=$(run_turn "$q" "$cid" "$uid") || {
            fail "E2E 3c — request '$q' timed out"; return
        }
        cid=$(echo "$resp" | jget "['conversation_id']" 2>/dev/null || echo "$cid")
    done

    sleep "$PROFILE_REFRESH_SEC"

    local default_team
    default_team=$(psql_q "
        SELECT preferences->>'default_team'
        FROM user_profiles
        WHERE external_id = '$uid';
    " | head -1)

    if [[ "$default_team" != "cthulhu" ]]; then
        fail "E2E 3c: default_team='$default_team', expected 'cthulhu'"
        return
    fi
    pass "E2E 3c: default_team='cthulhu' set after 6 entity-bearing messages"

    # Now: query without team — supervisor must inject default_team.
    local resp_inj cid_inj
    resp_inj=$(run_turn "Покажи velocity" "" "$uid") || {
        fail "E2E 3c — injection turn timed out"; return
    }
    cid_inj=$(echo "$resp_inj" | jget "['result']['conversation_id']" 2>/dev/null \
              || echo "")
    if [[ -z "$cid_inj" || "$cid_inj" == "None" ]]; then
        # Fallback: pull conv id from the create response if result.conversation_id
        # wasn't surfaced (shouldn't happen, but defensive).
        cid_inj=$(echo "$resp_inj" | jget "['conversation_id']" 2>/dev/null || echo "")
    fi
    sleep 2

    local injected_team
    injected_team=$(psql_q "
        SELECT metadata->'entities'->>'team_name'
        FROM messages
        WHERE conversation_id = '$cid_inj'
          AND role = 'user'
          AND turn_index = 0;
    " | head -1)

    if [[ "$injected_team" == "cthulhu" ]]; then
        pass "E2E 3c: 'Покажи velocity' got team_name='cthulhu' from profile"
    else
        fail "E2E 3c: 'Покажи velocity' got team_name='$injected_team', expected 'cthulhu'"
    fi
}

# 3d. Inactivity rotation: backdate updated_at, send another turn, verify
# rotation + audit-row repoint.
run_e2e_rotation() {
    note "E2E 3d: inactivity rotation + tasks.conversation_id repoint"
    local uid="${VERIFY_UID}-rot"

    local resp1 cid_old
    resp1=$(run_turn "Velocity cthulhu" "" "$uid") || {
        fail "E2E 3d — turn 1 timed out"; return
    }
    cid_old=$(echo "$resp1" | jget "['conversation_id']" 2>/dev/null || echo "")
    if [[ -z "$cid_old" ]]; then
        fail "E2E 3d — no conversation_id from turn 1"; return
    fi
    sleep 1

    # Force this conversation's updated_at past the 30-min threshold.
    psql_q "
        UPDATE conversations
        SET updated_at = NOW() - INTERVAL '31 minutes'
        WHERE id = '$cid_old';
    " >/dev/null || {
        fail "E2E 3d — could not backdate conversation"; return
    }

    local resp2 task2 cid_new
    resp2=$(run_turn "Что нового?" "$cid_old" "$uid") || {
        fail "E2E 3d — turn 2 timed out"; return
    }
    task2=$(echo "$resp2" | jget "['task_id']" 2>/dev/null || echo "")
    cid_new=$(echo "$resp2" | jget "['result']['conversation_id']" 2>/dev/null \
              || echo "")

    if [[ -z "$cid_new" || "$cid_new" == "None" || "$cid_new" == "$cid_old" ]]; then
        fail "E2E 3d: no rotation; result.conversation_id='$cid_new', original='$cid_old'"
        return
    fi
    pass "E2E 3d: rotated to fresh conversation ($cid_new)"

    local old_active
    old_active=$(psql_q "
        SELECT is_active FROM conversations WHERE id = '$cid_old';
    " | head -1)
    if [[ "$old_active" == "f" ]]; then
        pass "E2E 3d: old conversation marked is_active=false"
    else
        fail "E2E 3d: old is_active='$old_active', expected 'f'"
    fi

    # Audit-row repoint (Часть 1.2).
    local task_cid
    task_cid=$(psql_q "
        SELECT conversation_id FROM tasks WHERE task_id = '$task2';
    " | head -1)
    if [[ "$task_cid" == "$cid_new" ]]; then
        pass "E2E 3d: tasks.conversation_id repointed to new conversation"
    else
        fail "E2E 3d: tasks.conversation_id='$task_cid', expected '$cid_new'"
    fi
}

# 3e. Session restore: GET /conversations/{id}/messages round-trip.
run_e2e_restore() {
    note "E2E 3e: session restore via GET /conversations/{id}/messages"
    local uid="${VERIFY_UID}-restore"

    local resp cid
    resp=$(run_turn "Расскажи о задаче AL-38787" "" "$uid") || {
        fail "E2E 3e — request timed out"; return
    }
    cid=$(echo "$resp" | jget "['conversation_id']" 2>/dev/null || echo "")
    sleep 2

    local n
    n=$(docker compose exec -T api \
        curl -sS "http://localhost:8080/conversations/$cid/messages" \
        | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(len(data) if isinstance(data, list) else 0)
" 2>/dev/null || echo "0")

    if [[ "$n" -ge 2 ]]; then
        pass "E2E 3e: GET /conversations/$cid/messages returned $n messages"
    else
        fail "E2E 3e: returned $n messages (expected ≥ 2)"
    fi
}

run_e2e() {
    section "Level 3 — End-to-end (live API + DB)"
    if (( RUN_E2E == 0 )); then skip "E2E"; return; fi

    if (( HAVE_API == 0 || HAVE_POSTGRES == 0 || HAVE_VLLM == 0 \
          || HAVE_VLLM_SQL == 0 || HAVE_CELERY == 0 )); then
        skip "E2E — full stack required (api/postgres/vllm/vllm-sql/celery)"
        return
    fi
    if (( HAVE_QDRANT == 0 )); then
        note "qdrant not up — RAG branch will degrade, but our queries don't trigger RAG"
    fi

    run_e2e_short_term
    run_e2e_profile_gate
    run_e2e_profile_injection
    run_e2e_rotation
    run_e2e_restore
}

# ─────────────────────────────────────────────────────────────
# LEVEL 4 — Regression
# ─────────────────────────────────────────────────────────────

run_regression() {
    section "Level 4 — Regression evals"
    if (( RUN_REGRESSION == 0 )); then
        skip "regression — opt in with --regression"
        return
    fi
    if (( HAVE_VLLM == 0 || HAVE_POSTGRES == 0 )); then
        skip "regression — needs at least vllm + postgres"; return
    fi
    if (( HAVE_QDRANT == 0 || HAVE_VLLM_SQL == 0 )); then
        note "qdrant or vllm-sql missing — RAG/SQL evals will degrade"
    fi

    local stamp
    stamp="regress_$(date +%Y%m%d_%H%M%S)"
    note "running 4 baseline evals; this is slow (~10 min on RTX 4090)"

    if docker compose run --rm app \
        python -m eval.run_supervisor_eval --experiment "${stamp}_sup" \
        >> "$LOG_FILE" 2>&1; then
        pass "supervisor eval ran (results in eval/results/${stamp}_sup_*.json)"
    else
        fail "supervisor eval crashed — see $LOG_FILE"
    fi

    if docker compose run --rm app \
        python -m eval.run_sql_eval --experiment "${stamp}_sql" \
        >> "$LOG_FILE" 2>&1; then
        pass "sql eval ran"
    else
        fail "sql eval crashed — see $LOG_FILE"
    fi

    if docker compose run --rm app \
        python -m eval.run_eval --experiment "${stamp}_rag" \
        >> "$LOG_FILE" 2>&1; then
        pass "rag eval ran"
    else
        fail "rag eval crashed — see $LOG_FILE"
    fi

    if docker compose run --rm app \
        python -m eval.run_response_eval --experiment "${stamp}_resp" \
        >> "$LOG_FILE" 2>&1; then
        pass "response eval ran"
    else
        fail "response eval crashed — see $LOG_FILE"
    fi

    note "compare against baseline manually: python -m eval.compare"
}

# ─────────────────────────────────────────────────────────────
# Cleanup test data
# FK landscape (per migrations 003/004/005):
#   conversations.user_id  → user_profiles.id  ON DELETE SET NULL
#   messages.conversation_id → conversations.id ON DELETE CASCADE
#   conversation_summaries.conversation_id → conversations.id CASCADE
#   conversation_summaries.user_id → user_profiles.id CASCADE
#   tasks.conversation_id  → conversations.id ON DELETE SET NULL
# So: delete conversations first (cascades to messages + summaries),
#     then delete user_profiles. Tasks keep their rows but lose the FK
#     reference — that's acceptable for verification cleanup.
# ─────────────────────────────────────────────────────────────

cleanup_test_data() {
    section "Cleanup"
    if (( HAVE_POSTGRES == 0 )); then
        skip "cleanup — postgres not up"; return
    fi
    note "removing test rows with prefix '${VERIFY_UID}'"
    psql_q "
        DELETE FROM conversations
        WHERE user_id IN (
            SELECT id FROM user_profiles
            WHERE external_id LIKE '${VERIFY_UID}%'
        );
    " >/dev/null 2>&1 || true
    psql_q "
        DELETE FROM user_profiles WHERE external_id LIKE '${VERIFY_UID}%';
    " >/dev/null 2>&1 || true
    note "done"
}

# ─────────────────────────────────────────────────────────────
# Run all levels
# ─────────────────────────────────────────────────────────────

echo "Memory Layer verification"
echo "${DIM}log: $LOG_FILE${CLR}"
echo "${DIM}test prefix: $VERIFY_UID${CLR}"

run_unit_tests
run_multiturn_eval
run_e2e
run_regression
cleanup_test_data

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

section "Summary"
total=$(( PASS_COUNT + FAIL_COUNT ))
echo "  ${GRN}PASS${CLR}: $PASS_COUNT/$total"
if (( FAIL_COUNT > 0 )); then
    echo "  ${RED}FAIL${CLR}: $FAIL_COUNT/$total"
    echo
    echo "Failed checks:"
    for c in "${FAILED_CHECKS[@]}"; do
        echo "  - $c"
    done
    echo
    echo "Detailed logs: $LOG_FILE"
    exit 1
fi
echo
echo "${GRN}All enabled checks passed.${CLR}"
exit 0
