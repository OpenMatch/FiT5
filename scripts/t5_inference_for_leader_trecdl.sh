CP_DIR=/checkpoints_local/t5/
DATA_DIR=/dataset/trecdl2019
EXP_NAME=global_cocondenser_3layer
CP_NAME=global_cocondenser_3layer
STEP=1500
chk=/checkpoints_local/t5/global_cocondenser_3layer/global_cocondenser_3layer.bin
CUDA_VISIBLE_DEVICES=5 python  -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=11451  \
        ../inference_t5.py \
        -max_input 1280000 \
        -test $DATA_DIR/test_cocondenser_trecdl20_global.json  \
        -qrels /dataset/trecdl2019/test20_qrel \
        -checkpoint $chk \
        -res  $CP_DIR/$EXP_NAME/$CP_NAME-$STEP-test-trecdl20-for-leader.trec \
        -doc_size 100 \
        -log_dir $CP_DIR/$EXP_NAME/ \
        -test_batch_size 1 \
        -metric mrr_cut_10 \
        -max_query_len 64 \
        -max_seq_len 200 \
        -add_score \
        -add_bin \
        -number_bin \
        -use_global \
        -num_global_layers 3