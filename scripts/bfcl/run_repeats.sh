#!/usr/bin/env bash
# Run one BFCL A/B leg several times at once on one node and gate on the mean.
#
# When each arm needs only BFCL_TP GPUs, a node has room for several A/B pairs:
# run i puts arm A on GPUs [2i*TP, (2i+1)*TP) and arm B on the next TP GPUs,
# with its own ports, run dir and BFCL project root. The runs are scored
# concurrently, then `run_ab.py --combine` averages them and applies the usual
# gate to the mean, so the repeats cost GPUs rather than wall-clock time.
#
# Usage: run_repeats.sh <runs> <report.md> <report.json>
#
# Reads everything launch_arm.sh reads, plus:
#   BFCL_MODEL_FC     BFCL handler name (the `-FC` id)
#   CATEGORIES        comma-separated BFCL test categories
#   AB_ROOT           parent of the runs' BFCL project roots (run i: $AB_ROOT/run<i>)
#   BFCL_RUN_DIR      parent of the runs' pidfile/log dirs (run i: $BFCL_RUN_DIR/run<i>)
#   RUN_TIMEOUT       per-arm bfcl generate/evaluate cap in seconds (default 0 = none)
#   BFCL_PORT_BASE    first port; run i uses BASE+10i .. BASE+10i+3 (default 31000)
#   BFCL_TEMPERATURE  sampling temperature for run_ab.py (default: run_ab.py's)
#   BFCL_TOLERANCE    regression tolerance (default 0.02)
#   BFCL / PYTHON     bfcl executable and python (default: from PATH)
#
# Exit codes are run_ab.py's: 0 clean, 1 the mean regressed, 2 a run is incomplete.
set -euo pipefail

RUNS="${1:?usage: run_repeats.sh <runs> <report.md> <report.json>}"
OUT_MD="${2:?usage: run_repeats.sh <runs> <report.md> <report.json>}"
OUT_JSON="${3:?usage: run_repeats.sh <runs> <report.md> <report.json>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="${BFCL_TP:-1}"
AB_ROOT="${AB_ROOT:?set AB_ROOT}"
RUN_ROOT="${BFCL_RUN_DIR:?set BFCL_RUN_DIR}"
PORT_BASE="${BFCL_PORT_BASE:-31000}"
PYTHON="${PYTHON:-python}"
BFCL="${BFCL:-$(command -v bfcl)}"

stop_all() {
  for ((i = 0; i < RUNS; i++)); do
    BFCL_RUN_DIR="$RUN_ROOT/run$i" bash "$HERE/launch_arm.sh" stop || true
  done
}
trap stop_all EXIT

# launch <run> <a|b> <first GPU>: start one arm on TP GPUs with the run's pinned
# ports (concurrent launches would otherwise race for the same free port).
launch() {
  local i="$1" arm="$2" first="$3" base=$((PORT_BASE + 10 * $1))
  BFCL_RUN_DIR="$RUN_ROOT/run$i" BFCL_GPU="$(seq -s, "$first" $((first + TP - 1)))" \
    BFCL_ARM_A_PORT="$base" BFCL_ARM_B_GRPC_PORT=$((base + 1)) \
    BFCL_ARM_B_GW_PORT=$((base + 2)) BFCL_ARM_B_METRICS_PORT=$((base + 3)) \
    bash "$HERE/launch_arm.sh" "$arm" >"$RUN_ROOT/run$i/url_$arm" 2>"$RUN_ROOT/run$i/launch_$arm.log"
}

# 1) Bring up every run's two arms at once.
declare -A launching=()
for ((i = 0; i < RUNS; i++)); do
  mkdir -p "$RUN_ROOT/run$i" "$AB_ROOT/run$i"
  launch "$i" a $((2 * i * TP)) &
  launching[$!]="$i a"
  launch "$i" b $(((2 * i + 1) * TP)) &
  launching[$!]="$i b"
done
declare -A down=()
for pid in "${!launching[@]}"; do
  if ! wait "$pid"; then
    read -r i arm <<<"${launching[$pid]}"
    down[$i]=1
    echo "[run_repeats] run $i arm $arm failed to start; last lines of its launch log:" >&2
    tail -n 30 "$RUN_ROOT/run$i/launch_$arm.log" >&2 || true
  fi
done

# 2) Score the runs whose arms are up, concurrently; each writes its own report.
scoring=()
for ((i = 0; i < RUNS; i++)); do
  if [ -n "${down[$i]:-}" ]; then
    BFCL_RUN_DIR="$RUN_ROOT/run$i" bash "$HERE/launch_arm.sh" stop || true
    continue
  fi
  args=(
    --baseline "vllm=$(cat "$RUN_ROOT/run$i/url_a")"
    --candidate "smg=$(cat "$RUN_ROOT/run$i/url_b")"
    --bfcl-model "$BFCL_MODEL_FC" --categories "$CATEGORIES" --bfcl "$BFCL"
    --project-root "$AB_ROOT/run$i" --run-timeout "${RUN_TIMEOUT:-0}"
    --out "$AB_ROOT/run$i/bfcl_ab.md" --json-out "$AB_ROOT/run$i/bfcl_ab.json"
  )
  [ -n "${BFCL_TEMPERATURE:-}" ] && args+=(--temperature "$BFCL_TEMPERATURE")
  echo "[run_repeats] run $i: scoring (log: $AB_ROOT/run$i/run_ab.log)" >&2
  "$PYTHON" "$HERE/run_ab.py" "${args[@]}" >"$AB_ROOT/run$i/run_ab.log" 2>&1 &
  scoring+=($!)
done
# Each run's own gate is advisory; the combined report below decides.
for pid in "${scoring[@]}"; do wait "$pid" || true; done

# 3) Average the runs (a run without a report counts as incomplete) and gate.
reports=()
for ((i = 0; i < RUNS; i++)); do reports+=("$AB_ROOT/run$i/bfcl_ab.json"); done
rc=0
"$PYTHON" "$HERE/run_ab.py" --combine "${reports[@]}" --categories "$CATEGORIES" \
  --out "$OUT_MD" --json-out "$OUT_JSON" --tolerance "${BFCL_TOLERANCE:-0.02}" || rc=$?
exit "$rc"
