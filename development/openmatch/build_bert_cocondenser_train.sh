COLLECTION_DIR="/collections/"
PROCESSED_DIR="/dataset/msmarco/passage"
python scripts/msmarco/build_hn.py \
    --tokenizer_name bert-base-uncased \
    --hn_file /dataset/msmarco/passage/rank.tsv \
    --qrels $COLLECTION_DIR/marco/qrels.train.tsv \
    --queries $COLLECTION_DIR/marco/train.query.txt \
    --collection $COLLECTION_DIR/marco/corpus.tsv \
    --save_to $PROCESSED_DIR/openmatch/bert/ \
    --query_template "<text> [SEP]" \
    --doc_template "<title> <text>"