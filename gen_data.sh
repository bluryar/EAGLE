# python /root/EAGLE/eagle/data/gen.py  \
#   --dataset /root/EAGLE/eagle/data/raw/openo1.jsonl   \
#   --vllm_url http://127.0.0.1:30000/v1/chat/completions \
#   --output_dir /root/EAGLE/eagle/data/save \
#   --threads 32 \
#   --timeout 180 \
#   --batch_save 10000

python /root/EAGLE/eagle/data/distilabel_gen.py \
  --dataset /root/EAGLE/eagle/data/raw/openo1.jsonl \
  --vllm_url http://127.0.0.1:30000/v1 \
  --temperature 0.6 \
  --model_name /root/Qwen3-4B-FP8 \
  --max_new_tokens 2048 \
  --batch_size 1024 \
  --use_cache \
  --output_dir /root/EAGLE/eagle/data/save