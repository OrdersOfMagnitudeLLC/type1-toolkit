#!/bin/bash
# Orchestration script for ablation + LoRA KD on RunPod.
# Run inside tmux on the pod: tmux new-session -s pipeline 'bash /workspace/run_all.sh > /workspace/pipeline.log 2>&1'
set -e

echo "=== $(date) — Pipeline started ==="

# --- Config ---
QWEN05B="/workspace/qwen0.5b/master"
QWEN3B="/workspace/qwen3b"
CACHE_DIR="/workspace/teacher_logits_cache_qwen3b"
ABLATION_DATA="/workspace/ablation_data"
RESULTS_DIR="/workspace/results"
STEPS=2500
BATCH=4
LR=1e-5

mkdir -p "$RESULTS_DIR"

# --- Step 1: Prepare ablation data ---
echo "=== $(date) — Step 1: Prepare ablation data ==="
python3 /workspace/ablation_prepare.py \
    --tokenizer "$QWEN05B" \
    --output-dir "$ABLATION_DATA" \
    --n-samples 10000 \
    --retain-pct 0.13

# --- Step 2: Train standard (GPU) ---
echo "=== $(date) — Step 2: Train standard model ==="
python3 /workspace/ablation_train.py \
    --base-model "$QWEN05B" \
    --data "$ABLATION_DATA/standard_data.jsonl" \
    --output "$RESULTS_DIR/ckpt_standard" \
    --steps $STEPS --batch-size $BATCH --lr $LR

# --- Step 3: Train NS-filtered (GPU) ---
echo "=== $(date) — Step 3: Train NS-filtered model ==="
python3 /workspace/ablation_train.py \
    --base-model "$QWEN05B" \
    --data "$ABLATION_DATA/ns_filtered_data.jsonl" \
    --output "$RESULTS_DIR/ckpt_ns" \
    --steps $STEPS --batch-size $BATCH --lr $LR

# --- Step 4: Start eval (CPU, background) + LoRA KD (GPU, foreground) ---
echo "=== $(date) — Step 4a: Start eval on CPU (background) ==="
tmux kill-session -t eval 2>/dev/null || true
tmux new-session -d -s eval "bash -c '
    echo \"=== Eval standard ===\" > /workspace/eval.log
    lm_eval --model hf --model_args pretrained=$RESULTS_DIR/ckpt_standard,dtype=bfloat16 \
        --tasks hellaswag,arc_easy,arc_challenge,mmlu --device cpu --batch_size 4 \
        --output_path $RESULTS_DIR/eval_standard.json >> /workspace/eval.log 2>&1
    echo \"=== Eval NS-filtered ===\" >> /workspace/eval.log
    lm_eval --model hf --model_args pretrained=$RESULTS_DIR/ckpt_ns,dtype=bfloat16 \
        --tasks hellaswag,arc_easy,arc_challenge,mmlu --device cpu --batch_size 4 \
        --output_path $RESULTS_DIR/eval_ns.json >> /workspace/eval.log 2>&1
    echo \"EVAL_DONE\" >> /workspace/eval.log
'"

echo "=== $(date) — Step 4b: LoRA KD on Qwen 3B (GPU) ==="
python3 /workspace/train_lora_kd.py \
    --base-model "$QWEN3B" \
    --cache-dir "$CACHE_DIR" \
    --output "$RESULTS_DIR/oom_3b_lora" \
    --rank 16 --steps 2000 --batch-size 4 --lr 2e-4 \
    --target-modules "q_proj,v_proj"

echo "=== $(date) — LoRA KD done, waiting for eval ==="
while ! grep -q "EVAL_DONE" /workspace/eval.log 2>/dev/null; do
    sleep 30
    echo "  Eval still running... $(tail -1 /workspace/eval.log 2>/dev/null)"
done
echo "=== $(date) — Eval done ==="

# --- Step 5: Summary ---
echo "=== $(date) — Pipeline complete ==="
echo "=== Eval results: ==="
cat "$RESULTS_DIR"/eval_*.json 2>/dev/null | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line)
        for k, v in d.get('results', {}).items():
            print(f'  {k}: {v}')
    except: pass
"
echo "=== Files: ==="
ls -lhR "$RESULTS_DIR/"

echo "=== $(date) — ALL DONE ==="
