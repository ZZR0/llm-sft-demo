#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file scripts/zero3.yaml scripts/finetune.py \
    --model_name_or_path data/models/qwen2.5-base-3b \
    --train_data_path data/train.jsonl \
    --eval_data_path data/eval.jsonl \
    --output_dir output/qwen2.5_base_3b_sft \
    --model_max_length 4096 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --learning_rate 1e-5 \
    --bf16 true \
    --logging_dir output/qwen2.5_base_3b_sft/logs \
    --logging_steps 2 \
    --report_to none \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --seed 42