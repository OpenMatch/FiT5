CP_DIR=/checkpoints_local/t5/
DATA_DIR=/dataset/msmarco/passage
EXP_NAME=global_cocondenser_3layer
CP_NAME=global_cocondenser_3layer
STEP=1500
chk=/checkpoints_local/t5/global_cocondenser_3layer/global_cocondenser_3layer.bin
CUDA_VISIBLE_DEVICES=4,5,6,7 python  -m torch.distributed.launch \
        --nproc_per_node=4 \
        --master_port=11451  \
        /Global-NeuLTR-master/inference_t5.py \
        -max_input 1280000 \
        -test $DATA_DIR/test_cocondenser_mine_global.json  \
        -qrels $DATA_DIR/test_cocondenser_mine_global.trec \
        -checkpoint $chk \
        -res  $CP_DIR/$EXP_NAME/$CP_NAME-$STEP-test-for-leader.trec \
        -doc_size 100 \
        -log_dir $CP_DIR/$EXP_NAME/ \
        -test_batch_size 1 \
        -metric mrr_cut_10 \
        -max_query_len 64 \
        -max_seq_len 200 \
        -use_global \
        -num_global_layers 3 \
        -add_score \
        -add_bin \
        -number_bin
