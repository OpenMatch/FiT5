COLLECTION_DIR="/collections/"
PROCESSED_DIR="/dataset/msmarco/passage"
python scripts/msmarco/build_hn_score.py \
    --tokenizer_name /t5-base-scaled \
    --hn_file /dataset/msmarco/passage/rank.tsv \
    --qrels $COLLECTION_DIR/marco/qrels.train.tsv \
    --queries $COLLECTION_DIR/marco/train.query.txt \
    --collection $COLLECTION_DIR/marco/corpus.tsv \
    --save_to $PROCESSED_DIR/openmatch_score \
    --query_template "Query: <text>" \
    --doc_template "Title: <title> score: <score> Document: <text> Relevant: "