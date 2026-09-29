# clipcheck.py (pang_reference, CPU-only diagnostic): runs feder_runner.py unmodified via runpy, with
# torch.nn.utils.clip_grad_norm_ wrapped to log every call: norm type, pre-clip total norm, bound, and
# whether it scaled the gradients (torch scales iff total_norm > max_norm, by max_norm/(total_norm+1e-6)).
import runpy, sys, torch
_orig = torch.nn.utils.clip_grad_norm_
calls = []
def logged(parameters, max_norm, norm_type=2.0, *a, **k):
    params = [p for p in parameters if p.grad is not None]
    total = _orig(params, max_norm, norm_type, *a, **k)
    t = float(total)
    calls.append((str(norm_type), t))
    print(f"[clipcheck] call {len(calls)} norm_type={norm_type} total_norm={t:.6g} max_norm={max_norm} clipped={t > max_norm}", flush=True)
    return total
torch.nn.utils.clip_grad_norm_ = logged
# The burn-in checkpoint was saved on the GPU; load it on the CPU (CUDA is hidden for this check)
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{'map_location': 'cpu', **k})
sys.argv = ['feder_runner.py'] + sys.argv[1:]
runpy.run_path('feder_runner.py', run_name='__main__')
