COLLECTION_DIR="/collections/"
PROCESSED_DIR="/dataset/msmarco/passage"
python scripts/msmarco/build_hn.py \
    --tokenizer_name /t5-base-scaled \
    --hn_file /OpenMatch/experiments/inference.cocodr-base-msmarco.msmarco-train/test.trec \
    --qrels $COLLECTION_DIR/marco/qrels.train.tsv \
    --queries $COLLECTION_DIR/marco/train.query.txt \
    --collection $COLLECTION_DIR/marco/corpus.tsv \
    --save_to $PROCESSED_DIR/openmatch \
    --query_template "Query: <text>" \
    --doc_template "Document: <text> Relevant: "