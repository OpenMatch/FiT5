CHECKPOINT_DIR=/OpenMatch/checkpoints/msmarco_condenser_all_title_score3/
PROCESSED_DIR="/dataset/msmarco/passage/openmatch"
LOG_DIR="/OpenMatch/logs"
#/t5-base-scaled
#/dataset/msmarco/passage/openmatch
#/openmatch_data/processed/msmarco/t5_cocondenser/train.jsonl
title_path="/OpenMatch/checkpoints/msmarco_condenser_all_title/checkpoint-100000"

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=11455 -m openmatch.driver.train_rr  \
    --output_dir $CHECKPOINT_DIR  \
    --model_name_or_path $title_path \
    --fp16 \
    --do_train  \
    --save_steps 500  \
    --train_path /dataset/msmarco/passage/openmatch_score/train.new.jsonl  \
    --eval_path /dataset/msmarco/passage/openmatch_score/val.jsonl  \
    --eval_steps 500  \
    --per_device_train_batch_size 64  \
    --per_device_eval_batch_size 128  \
    --gradient_accumulation_steps 128 \
    --learning_rate 1e-5  \
    --q_max_len 32  \
    --p_max_len 166  \
    --num_train_epochs 150  \
    --pos_token true  \
    --neg_token false  \
    --warmup_steps 1000 \
    --logging_dir $LOG_DIR/t5  \
    --evaluation_strategy steps  \
    --dataloader_num_workers 1