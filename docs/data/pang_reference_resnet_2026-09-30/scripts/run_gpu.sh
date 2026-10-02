#!/bin/bash
# The authors' federated setup on the GPU (pang_reference). Started by queue_pangref.sh only after EXP-LLM-004.
# Everything logs to /home/soham/HDD/pang_reference/runs/ (per-run .out) plus the reference Logger files
# in checkpoints_base_bn/ (also on the share).
set -u
cd /home/soham/Soham/pang_reference/AccumulativeAttack
PY=/home/soham/Soham/pang_reference/.venv/bin/python
RUNS=/home/soham/HDD/pang_reference/runs
log() { echo "[$(date '+%F %T')] $*"; }
gpu_idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
run() {  # name, args...
  local name=$1; shift
  if [ -f "$RUNS/$name.done" ]; then log "skip $name (done)"; return; fi
  while ! gpu_idle; do log "GPU busy with another process; waiting before $name"; sleep 120; done
  log "start $name: $*"
  "$PY" feder_runner.py "$@" --log_name "log_$name.txt" > "$RUNS/$name.out" 2>&1 && touch "$RUNS/$name.done"
  log "end $name (exit $?): $(grep -E 'Test error (before|after) tri|after poisoned tri' "$RUNS/$name.out" | tr '\n' ' ')"
}

# (a) burn-in as the reference does: train_cifar.py, 40 epochs, batch 256, lr 0.1, SGD momentum 0.9, wd 1e-4, BN
if [ ! -f checkpoints_base_bn/epoch40.pth ]; then
  while ! gpu_idle; do log "GPU busy; waiting before burn-in"; sleep 120; done
  log "burn-in start"
  "$PY" train_cifar.py --outf checkpoints_base_bn --workers 8 > "$RUNS/burnin.out" 2>&1
  log "burn-in end (exit $?): $(grep 'err_cls' "$RUNS/burnin.out" | tail -1)"
fi
[ -f checkpoints_base_bn/epoch40.pth ] || { log "no burn-in checkpoint; stopping"; exit 1; }

COMMON="--batch_size 100 --test_batch_size 500 --resume checkpoints_base_bn --use_bn --model_name epoch40.pth --mode train --onlinemode train --lr 0.1 --momentum 0.9"
for seed in 1 2 3; do
  # (c) baselines: the same clean trigger without accumulation; Table 3's direct poisoned trigger
  run "noaccum_seed$seed" --main federated --epochs 0 --seed $seed $COMMON
  for s in 1 10 20 50; do run "direct_s${s}_seed$seed" --main direct --feder_lambda $s --seed $seed $COMMON; done
  # (b) accumulative phase (1000 steps) + clean trigger, Table 3 no-clip column
  for lam in 0.01 0.02 0.05 0.08; do run "accu_l${lam}_seed$seed" --main federated --epochs 1000 --feder_lambda $lam --seed $seed $COMMON; done
done
# If time allows: Table 3's l2-clip columns (clip applies to the accumulative updates; the trigger step clips l-inf)
for seed in 1 2 3; do for c in 10 1; do
  run "accu_l0.05_clip${c}_seed$seed" --main federated --epochs 1000 --feder_lambda 0.05 --clip_gradnorm --clipvalue $c --seed $seed $COMMON
  run "direct_s50_clip${c}_seed$seed" --main direct --feder_lambda 50 --clip_gradnorm --clipvalue $c --seed $seed $COMMON
done; done
log "all runs finished"
