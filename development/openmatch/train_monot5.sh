CHECKPOINT_DIR=/OpenMatch/checkpoints/msmarco_bm25_test_3/
PROCESSED_DIR="/dataset/msmarco/passage/openmatch"
LOG_DIR="/OpenMatch/logs"
#/t5-base-scaled
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=11455 -m openmatch.driver.train_rr  \
    --output_dir $CHECKPOINT_DIR  \
    --model_name_or_path t5-base\
    --do_train  \
    --save_steps 10000  \
    --train_path /dataset/msmarco/passage/openmatch/train.new.jsonl  \
    --eval_path /dataset/msmarco/passage/openmatch/val.jsonl  \
    --eval_steps 10000  \
    --per_device_train_batch_size 16  \
    --per_device_eval_batch_size 128  \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-5  \
    --q_max_len 64  \
    --p_max_len 446  \
    --num_train_epochs 16  \
    --pos_token true  \
    --neg_token false  \
    --adafactor \
    --weight_decay 5e-5 \
    --warmup_steps 1000 \
    --logging_dir $LOG_DIR/t5  \
    --evaluation_strategy steps  \
    --dataloader_num_workers 1