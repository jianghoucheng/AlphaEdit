python /mnt/afs/wwh/Work/nlp/AlphaEdit/experiments/evaluate.py \
    --alg_name AlphaEdit \
    --model_name /mnt/afs/models/Qwen/Qwen3.5-27B \
    --hparams_fname qwen3.5-27b.json \
    --dataset_size_limit 100 \
    --skip_generation_tests \
    --use_cache \
    --output_dir /mnt/afs/wwh/Work/nlp/AlphaEdit/experiments/results 