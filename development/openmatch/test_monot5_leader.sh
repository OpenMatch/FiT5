COLLECTION_DIR="/collections/"
PROCESSED_DIR="/dataset/msmarco/passage/openmatch"
LOG_DIR="/OpenMatch/logs"
STEP=100000
CHECKPOINT_DIR="/OpenMatch/checkpoints/msmarco_condenser_all_title/checkpoint-$STEP"
chk2="/pygaggle/pygaggle/checkpoint/checkpoint-10000/"
#/msmarco-100k
cp /t5-base-scaled/spiece.model $CHECKPOINT_DIR
CUDA_VISIBLE_DEVICES=6 python -m openmatch.driver.rerank  \
    --output_dir None  \
    --model_name_or_path  $CHECKPOINT_DIR\
    --per_device_eval_batch_size 256  \
    --query_path /dataset/msmarco/passage/docleaderboard-queries.tsv  \
    --corpus_path $COLLECTION_DIR/marco/corpus.tsv  \
    --query_template "Query: <text>" \
    --doc_template "Document: <title> <text> Relevant: " \
    --query_column_names id,text  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 166  \
    --pos_token true  \
    --neg_token false  \
    --fp16  \
    --trec_run_path  /dataset/msmarco/passage/test_rank.tsv \
    --trec_save_path /OpenMatch/results/condenser-rr-test-title-leader-$STEP.trec  \
    --dataloader_num_workers 1 \
    --reranking_depth 100