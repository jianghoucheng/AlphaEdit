from pathlib import Path
from transformers import AutoConfig
import os

STATS_DIR = Path("data/stats")
model_name_arg = "meta-llama/Meta-Llama-3-8B-Instruct"

# Simulate model config
class Config:
    _name_or_path = model_name_arg

config = Config()

short_name = config._name_or_path.rsplit("/")[-1]
full_name = config._name_or_path.replace("/", "_")

print(f"Short: {short_name}")
print(f"Full: {full_name}")

if (STATS_DIR / full_name).exists():
    model_name = full_name
    print(f"Chosen model_name: {model_name} (found full)")
else:
    model_name = short_name
    print(f"Chosen model_name: {model_name} (fallback short)")

ds_name = "wikipedia"
layer_name = "model.layers.4.mlp.down_proj"
precision = "float32"
to_collect = ["mom2"]
sample_size = 100000

# Batch tokens logic
# Assume defaults
npos = 8192 # Llama 3
batch_tokens = npos * 3 

size_suffix = "" if sample_size is None else f"_{sample_size}"
if batch_tokens < npos:
    size_suffix = "_t{batch_tokens}" + size_suffix

file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
filename = STATS_DIR / file_extension

print(f"Constructed filename: {filename}")
print(f"Exists? {filename.exists()}")

if not filename.exists():
    print("Files in dir:")
    parent_dir = filename.parent
    if parent_dir.exists():
        print(os.listdir(parent_dir))
    else:
        print(f"Parent dir {parent_dir} does not exist")
