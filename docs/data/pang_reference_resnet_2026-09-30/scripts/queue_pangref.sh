#!/bin/bash
# Queue for the pang_reference GPU runs (tmux `pangref`). Starts run_gpu.sh only when EXP-LLM-004 has finished:
#   (1) its log has a compile_results MEAN row, (2) no process of it remains, (3) the GPU has no compute process.
set -u
LOG4=/home/soham/Soham/Poisoning_LLM/experiments/polarity_3b_baseline/run.log
QLOG4=/home/soham/Soham/Poisoning_LLM/experiments/polarity_3b_baseline/queue.log
log() { echo "[$(date '+%F %T')] $*"; }
done_004() { [ -f "$LOG4" ] && tr '\r' '\n' < "$LOG4" | grep -qE '^MEAN [0-9]'; }
alive_004() { pgrep -f "polarity_3b_baseline" > /dev/null || pgrep -f "queue_after_003.sh" > /dev/null; }
gpu_idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }

log "waiting for EXP-LLM-004 (MEAN row in $LOG4, no 004 process, idle GPU)"
warned=0; n=0
while true; do
  if done_004 && ! alive_004 && gpu_idle; then break; fi
  # Only the latest line counts: a restarted queue appends to the same log
  if [ $warned -eq 0 ] && [ -f "$QLOG4" ] && tail -1 "$QLOG4" | grep -q "NOT starting EXP-LLM-004"; then
    log "NOTE: EXP-LLM-004's queue declined to start it ($(tail -1 "$QLOG4")). Still waiting; 004 must be run and finish first."
    warned=1
  fi
  n=$((n + 1)); if [ $((n % 30)) -eq 0 ]; then log "still waiting: MEAN=$(done_004 && echo yes || echo no) 004-alive=$(alive_004 && echo yes || echo no) gpu-idle=$(gpu_idle && echo yes || echo no)"; fi
  sleep 60
done
log "EXP-LLM-004 finished and the GPU is idle; starting in 60 s"
sleep 60
bash /home/soham/Soham/pang_reference/run_gpu.sh
