#!/usr/bin/env bash
# Start the per-chunk context run (eval/run_blurb_eval.py --granularity chunk)
# at a given time today, with LM Studio loaded the way the run needs:
# 32k context and one parallel slot (with more slots the server stops reusing
# each note's prompt prefix and every call re-reads the whole article).
#
# Usage, detached and keeping the Mac awake until the run ends:
#   nohup caffeinate -is eval/run_chunk_context_overnight.sh 20:00 \
#     > eval/results/logs/chunk-context-launch.out 2>&1 & disown
#
# The run's own output goes to eval/results/logs/chunk-context-run.out.
set -euo pipefail
cd "$(dirname "$0")/.."

AT="${1:-20:00}"
MODEL="${ARIOSTEA_CTX_MODEL:-qwen2.5-14b-instruct-mlx}"
RUN_LOG=eval/results/logs/chunk-context-run.out

# Fail now, not at launch time.
for tool in lms uv; do
  command -v "$tool" >/dev/null || { echo "$tool is not on PATH"; exit 1; }
done
target=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%F) $AT" +%s)
if [ "$target" -le "$(date +%s)" ]; then
  echo "$AT has already passed today"; exit 1
fi

echo "$(date '+%F %T') waiting until $AT"
# Compare against the clock rather than sleeping for a fixed span: a sleep's
# countdown pauses while the machine sleeps.
while [ "$(date +%s)" -lt "$target" ]; do sleep 30; done

echo "$(date '+%F %T') loading $MODEL"
lms server start >/dev/null 2>&1 || true
lms unload --all >/dev/null 2>&1 || true
lms load "$MODEL" --context-length 32768 --parallel 1 -y
if ! lms ps | grep -E "^$MODEL[[:space:]]" | grep -Eq "[[:space:]]32768[[:space:]]+1[[:space:]]"; then
  echo "$MODEL is not loaded with a 32768 context and 1 parallel slot:"; lms ps; exit 1
fi

echo "$(date '+%F %T') starting the run, output in $RUN_LOG"
status=0
uv run python eval/run_blurb_eval.py --granularity chunk > "$RUN_LOG" 2>&1 || status=$?
echo "$(date '+%F %T') run exited with status $status"
lms unload --all >/dev/null 2>&1 || true
exit "$status"
