#!/bin/bash
# PyTorch/XLA version of run_roberta_lora.sh for a TPU v5e-4 host (4 chips).
# Differences from the GPU script:
#  - launched via src/xla_spawn.py (one process per TPU chip) with PJRT_DEVICE=TPU
#  - per-device batch 8 x 4 chips = global batch 32 (same as the GPU run); eval 32 x 4 = 128
#  - no --fp16 (CUDA AMP is not available on TPU). We also do NOT pass --bf16, because
#    run_glue.py would then load the weights themselves in bf16. Weights stay fp32;
#    XLA on TPU runs fp32 matmuls with bf16 passes by default.

# Environment setup
export TOKENIZERS_PARALLELISM=false
export PJRT_DEVICE=TPU

# Experiment configuration
task=mnli
exp_name=roberta_lora_${task}_xla
lr=5e-5
lr_ratio=80
num_chips=4

# Execute command
python src/xla_spawn.py \
  --model_name_or_path roberta-base \
  --task_name $task \
  --use_lora \
  --target_modules "query, value" \
  --do_train \
  --per_device_train_batch_size $((32 / num_chips)) \
  --per_device_eval_batch_size $((128 / num_chips)) \
  --max_seq_length 128 \
  --eval_steps 500 \
  --save_steps 500 \
  --logging_steps 10 \
  --num_train_epochs 3 \
  --learning_rate $lr \
  --loraplus_lr_ratio $lr_ratio \
  --lora_rank 8 \
  --lora_alpha 8 \
  --lr_scheduler_type 'linear' \
  --adam_beta1 0.9 \
  --adam_beta2 0.99 \
  --adam_epsilon 1e-8 \
  --output_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio} \
  --logging_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio}/logs/ \
  --evaluation_strategy steps \
  --save_strategy steps \
  --report_to tensorboard \
  --keep_checkpoints eval \
  --overwrite_output_dir \
  --ignore_mismatched_sizes \
  --save_total_limit 1 \
  "$@"
