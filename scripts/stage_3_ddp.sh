#!/bin/bash
export STORE_DIR=XXX

python -m torch.distributed.launch --nproc_per_node=8 pretrain_sc.py \
    --model_path $STORE_DIR/results/stage-2 \
    --use_linear_mask True \
    --fronzen False \
    --cross True \
    --data_path $STORE_DIR/training_dataset \
    --seed 2021 \
    --max_input_length 256 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.01 \
    --save_steps 10000 \
    --logging_steps 20 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 8 \
    --max_steps 100000 \
    --output_dir $STORE_DIR/results/stage-3 \
    --fp16 True
