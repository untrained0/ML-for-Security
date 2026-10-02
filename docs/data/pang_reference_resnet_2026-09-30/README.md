# Pang 2021 federated setting — the authors' reference code with ResNet-18 (2026-09-30)

Raw logs of the GPU runs behind the 2026-09-30 entry in `docs/memory.md`.

- Reference code: github.com/ShawnXYang/AccumulativeAttack @ 3aa3147, cloned at `/home/soham/Soham/pang_reference/AccumulativeAttack`.
- `scripts/feder_runner.py`: `feder_accu_train.py` verbatim except `.cuda()` → `.to(device)` and a `--main` selector
  (`federated` = Table 3's accumulative phase + clean trigger, `--epochs 0` = the same trigger without accumulation;
  `direct` = Table 3's direct poisoned trigger, loss scaling = `--feder_lambda`). The upstream file has CRLF endings; this
  copy keeps them on the original lines.
- `scripts/run_gpu.sh`: the exact run list (burn-in with the unmodified `train_cifar.py`, then 3 seeds × configs);
  `scripts/queue_pangref.sh`: the queue that waited for EXP-LLM-004.
- `scripts/clipcheck.py`: CPU-only diagnostic — runs `feder_runner.py` unmodified with `clip_grad_norm_` wrapped to log
  every call (norm type, pre-clip norm, bound) and `torch.load` mapped to the CPU.
- `runs/<config>_seed<n>.out`: stdout of each run (the `tensor(…)` lines in the clip runs are the pre-clip ℓ2 norms of
  the 1000 accumulative updates, printed by `craft_federated_NEW`). `runs/burnin.out`: the 40-epoch burn-in.
- Not copied (on the share): checkpoints `/home/soham/HDD/pang_reference/runs/checkpoints_base_bn/epoch{10,20,30,40}.pth`
  (342 MB) and the reference Logger files `log_*.txt` next to them. Environment: Python 3.10 venv at
  `/home/soham/Soham/pang_reference/.venv`, torch 2.5.1+cu124 / torchvision 0.20.1 installed on the share.
