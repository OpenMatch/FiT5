NAME=global_cocondenser_3layer
DATA_DIR=/dataset/msmarco/passage
SAVE_DIR=/checkpoints_local/t5/$NAME
mkdir -p $SAVE_DIR

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=11459  \
    train_t5_fp16_ddp_two_dev.py \
    -pretrained /t5-base-fp16.bin \
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
    -epoch 6 \
    -batch_size 1 \
    -loss CE \
    -lr 2e-5 \
    -use_global \
    -num_global_layers 3 \
    -gradient_accumulation_steps 4 \
    -doc_size 100 \
    -eval_every 5000 \
    -logging_step 500 \
    -metric mrr_cut_10 \
    -max_seq_len 200

