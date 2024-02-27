COLLECTION_DIR="/collections/"
PROCESSED_DIR="/dataset/msmarco/passage/openmatch"
LOG_DIR="/OpenMatch/logs"
STEP=50000
CHECKPOINT_DIR="/OpenMatch/checkpoints/bert_condenser_all_title/checkpoint-$STEP"
chk2="/pygaggle/pygaggle/checkpoint/checkpoint-10000/"
#/msmarco-100k
cp /OpenMatch/checkpoints/bert_condenser_all_title/spiece.model $CHECKPOINT_DIR
cp /OpenMatch/checkpoints/bert_condenser_all_title/vocab.txt $CHECKPOINT_DIR
cp /OpenMatch/checkpoints/bert_condenser_all_title/tokenizer_config.json $CHECKPOINT_DIR
#/dataset/trecdl2019/msmarco-test2019-queries.tsv #$COLLECTION_DIR/marco/dev.query.txt  \
#/dataset/trecdl2019/test_trecdl20_rank.tsv \  #/runs/run.msmarco-passage.cocondenser.txt \
#/OpenMatch/results/condenser-rr-test-$STEP.trec  \
CUDA_VISIBLE_DEVICES=5 python -m openmatch.driver.rerank  \
    --output_dir None  \
    --model_name_or_path  $CHECKPOINT_DIR\
    --per_device_eval_batch_size 512  \
    --query_path /dataset/trecdl2019/msmarco-test2019-queries.tsv \
    --corpus_path $COLLECTION_DIR/marco/corpus.tsv  \
    --query_template "<text> [SEP]" \
    --doc_template "<title> <text>" \
    --query_column_names id,text  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 166  \
    --pos_token true  \
    --neg_token false  \
    --fp16  \
    --trec_run_path  /dataset/trecdl2019/test_trecdl19_rank.tsv \
    --trec_save_path /OpenMatch/results/condenser-rr-bert-test-title-trecdl19-$STEP.trec  \
    --dataloader_num_workers 1 \
    --reranking_depth 100