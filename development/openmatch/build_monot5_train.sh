COLLECTION_DIR="/collections/"
PROCESSED_DIR="/dataset/msmarco/passage"
python scripts/msmarco/build_train.py \
    --tokenizer_name /t5-base-scaled  \
    --negative_file /monoT5/dataset/msmarco/train.negatives.tsv  \
    --qrels $COLLECTION_DIR/marco/qrels.train.tsv  \
    --queries $COLLECTION_DIR/marco/train.query.txt  \
    --collection $COLLECTION_DIR/marco/corpus.tsv  \
    --save_to $PROCESSED_DIR/openmatch/  \
    --query_template "Query: <text>" \
    --doc_template "Document: <text> Relevant: "