NAME=global_cocondenser_3layer
DATA_DIR=/dataset/msmarco/passage
SAVE_DIR=/checkpoints_local/t5/$NAME
chk=/checkpoints_local/t5/global_cocondenser_3layer/global_cocondenser_3layer.bin
mkdir -p $SAVE_DIR

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=11446  \
    train_t5_fp16_ddp_two_dev.py \
    -pretrained $chk2 \
    -config t5-base \
    -tokenizer t5-base \
    -train $DATA_DIR/train_cocondenser_495000_global.json \
    -dev  $DATA_DIR/train_cocondenser_dev_3195_global.json \
    -test $DATA_DIR/dev_cocondenser_mine_global.json \
    -qrels $DATA_DIR/train_cocondenser_dev_3195_global.trec \
    -qrels_test $DATA_DIR/qrels.dev.small.tsv \
    -save $SAVE_DIR/$NAME.bin \
    -log_dir $SAVE_DIR/ \
    -res $SAVE_DIR/$NAME.trec \
    -res_test $SAVE_DIR/$NAME_test.trec \
    -epoch 10 \
    -batch_size 1 \
    -loss CE \
    -lr 2e-5 \
    -gradient_accumulation_steps 64 \
    -doc_size 100 \
    -eval_every 500 \
    -logging_step 50 \
    -metric mrr_cut_10 \
    -max_seq_len 200 \
    -add_score \
    -add_bin \
    -retraining \
    -number_bin \
    -n_warmup_steps 1000 \
    -use_global \
    -num_global_layers 3


