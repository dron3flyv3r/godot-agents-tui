#!/usr/bin/env bash
set -euo pipefail

# Simple helper to run the raw RLlib interface against a checkpoint and capture output.
# Adjust these defaults as needed before running.

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/kasper/GameProjects/tanks!-agent/logs/rllib/PPO_2025-12-01_01-24-44/PPO_godot_22c09_00000_0_2025-12-01_01-24-44}"
CHECKPOINT_NUM="${CHECKPOINT_NUM:-82}"
MODE="${MODE:-multi}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
SPEEDUP="${SPEEDUP:-30}"
POLICY_ID="${POLICY_ID:-}"
LOG_FILE="${LOG_FILE:-/tmp/interface_run.log}"

PY="${PY:-/home/kasper/GameProjects/agents/.venv/bin/python}"
SCRIPT="${SCRIPT:-interface_rllib_raw.py}"

echo "Writing interface output to: ${LOG_FILE}"
echo "Checkpoint: ${CHECKPOINT_DIR} (number: ${CHECKPOINT_NUM})"
echo "Mode: ${MODE}, action_repeat: ${ACTION_REPEAT}, speedup: ${SPEEDUP}"
echo "Policy override: ${POLICY_ID:-<default>}"
echo "Starting in 2 seconds... (launch Godot editor now)"
sleep 2

cmd=(
    "${PY}"
    "${SCRIPT}"
    "${CHECKPOINT_DIR}"
    "--checkpoint-number=${CHECKPOINT_NUM}"
    "--mode=${MODE}"
    "--action-repeat=${ACTION_REPEAT}"
    "--speedup=${SPEEDUP}"
    "--log-tracebacks"
)

if [[ -n "${POLICY_ID}" ]]; then
    cmd+=("--policy=${POLICY_ID}")
fi

(
    echo "===== $(date --iso-8601=seconds) interface run start ====="
    echo "CMD: ${cmd[*]}"
    "${cmd[@]}"
) | tee "${LOG_FILE}"

echo "===== $(date --iso-8601=seconds) interface run finished =====" | tee -a "${LOG_FILE}"
