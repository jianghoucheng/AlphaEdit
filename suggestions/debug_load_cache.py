import sys
print("Starting...", flush=True)
try:
    from util.runningstats import load_cached_state
    import numpy as np
except Exception as e:
    print(f"Import failed: {e}", flush=True)
    sys.exit(1)

cachefile = "data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz"

args = {"sample_size": 100000}

print(f"Attempting to load {cachefile} with args {args}", flush=True)

try:
    res = load_cached_state(cachefile, args, quiet=False)
except Exception as e:
    print(f"load_cached_state raised {e}", flush=True)
    res = None

if res is None:
    print("Result is None", flush=True)
else:
    print("Result loaded successfully", flush=True)
    if "sample_size" in res:
        print(f"Loaded sample_size: {res['sample_size']}", flush=True)
