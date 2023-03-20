#!/bin/bash
export STORE_DIR=XXX
# trainer of transformers will automatically call all 
# available GPUs for data paralle training
export CUDA_VISIBLE_DEVICES=0

python pretrain_logic.py \
    --model_path $STORE_DIR/results/stage-1 \
    --use_linear_mask True \
    --data_path $STORE_DIR/training_dataset\
    --seed 2021 \
    --max_input_length 256 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.01 \
    --save_steps 10000 \
    --logging_steps 20 \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 16 \
    --max_steps 100000 \
    --output_dir $STORE_DIR/results/stage-2 \
    --fp16 True
