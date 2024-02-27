CHECKPOINT_DIR=/OpenMatch/checkpoints/msmarco_condenser_all_title/
PROCESSED_DIR="/dataset/msmarco/passage/openmatch"
LOG_DIR="/OpenMatch/logs"
#/t5-base-scaled
#/dataset/msmarco/passage/openmatch
#/openmatch_data/processed/msmarco/t5_cocondenser/train.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=11455 -m openmatch.driver.train_rr  \
    --output_dir $CHECKPOINT_DIR  \
    --model_name_or_path /t5-base-scaled\
    --fp16 \
    --do_train  \
    --save_steps 10000  \
    --train_path /dataset/msmarco/passage/openmatch/train.new.jsonl  \
    --eval_path /dataset/msmarco/passage/openmatch/val.jsonl  \
    --eval_steps 10000  \
    --per_device_train_batch_size 8  \
    --per_device_eval_batch_size 128  \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5  \
    --q_max_len 32  \
    --p_max_len 166  \
    --num_train_epochs 15  \
    --pos_token true  \
    --neg_token false  \
    --warmup_steps 1000 \
    --logging_dir $LOG_DIR/t5  \
    --evaluation_strategy steps  \
    --dataloader_num_workers 1