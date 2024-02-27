COLLECTION_DIR="/collections/"
PROCESSED_DIR="/dataset/msmarco/passage/openmatch"
LOG_DIR="/OpenMatch/logs"
STEPs=(1000 2000 3000 5000 6000 7000 8000 9000 10000)
for STEP in ${STEPs[@]}
do
#STEP=4000
echo $STEP
CHECKPOINT_DIR="/OpenMatch/checkpoints/msmarco_merge_2/checkpoint-$STEP"
cp /t5-base-scaled/spiece.model $CHECKPOINT_DIR
task="nfcorpus"
chk=$CHECKPOINT_DIR #"/msmarco-10k" #"/pygaggle/pygaggle/checkpoint2/checkpoint-10000/" #"/OpenMatch/checkpoints/msmarco_bm25_4/checkpoint-25000/" #"/OpenMatch/checkpoints/msmarco_bm25/checkpoint-80000/" #"/msmarco-100k"
dataset_name_list=(trec-covid nfcorpus fiqa arguana webis-touche2020 quora scidocs scifact nq hotpotqa signal1m trec-news robust04 dbpedia-entity fever climate-fever bioasq msmarco android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress) 
#dataset_name_list=(scifact trec-covid nfcorpus webis-touche2020 signal1m trec-news robust04) #(trec-covid fiqa arguana webis-touche2020 quora scidocs scifact nq hotpotqa signal1m trec-news robust04 dbpedia-entity fever) 
cp /t5-base-scaled/spiece.model $chk
name_prefix="merge-$STEP-rr"
TOT_CUDA="4,6"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="19876"
for task in ${dataset_name_list[@]}
do
    if [ ${task} == fiqa ] || [ ${task} == signal1m ]
    then
        export q_max_len=64
        export p_max_len=128
        echo ${task}
        CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} -m openmatch.driver.rerank  \
            --output_dir None  \
            --model_name_or_path $chk\
            --per_device_eval_batch_size 256  \
            --query_path /dataset/beir/$task/queries.jsonl  \
            --corpus_path /dataset/beir/$task/corpus.jsonl  \
            --query_template "Query: <text>" \
            --doc_template "Document: <text> Relevant: " \
            --query_column_names id,text  \
            --doc_column_names id,title,text  \
            --q_max_len $q_max_len  \
            --p_max_len $p_max_len  \
            --pos_token true  \
            --neg_token false  \
            --fp16  \
            --trec_run_path  /OpenMatch/experiments/inference.cocodr-base-msmarco.$task/test.trec \
            --trec_save_path /OpenMatch/results/beir/$name_prefix-$task.trec  \
            --dataloader_num_workers 1 \
            --reranking_depth 100
    elif [ ${task} == arguana ]
    then
        export q_max_len=128
        export p_max_len=128
        echo ${task}
        CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} -m openmatch.driver.rerank  \
            --output_dir None  \
            --model_name_or_path $chk\
            --per_device_eval_batch_size 256  \
            --query_path /dataset/beir/$task/queries.jsonl  \
            --corpus_path /dataset/beir/$task/corpus.jsonl  \
            --query_template "Query: <text>" \
            --doc_template "Document: <title> <text> Relevant: " \
            --query_column_names id,text  \
            --doc_column_names id,title,text  \
            --q_max_len $q_max_len  \
            --p_max_len $p_max_len  \
            --pos_token true  \
            --neg_token false  \
            --fp16  \
            --trec_run_path  /OpenMatch/experiments/inference.cocodr-base-msmarco.$task/test.trec \
            --trec_save_path /OpenMatch/results/beir/$name_prefix-$task.trec  \
            --dataloader_num_workers 1 \
            --reranking_depth 100
    elif [ ${task} == quora ]
    then
        export q_max_len=64
        export p_max_len=128
        echo ${task}
        CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} -m openmatch.driver.rerank  \
            --output_dir None  \
            --model_name_or_path $chk\
            --per_device_eval_batch_size 256  \
            --query_path /dataset/beir/$task/queries.jsonl  \
            --corpus_path /dataset/beir/$task/corpus.jsonl  \
            --query_template "Query: <text>" \
            --doc_template "Document: <text> Relevant: " \
            --query_column_names id,text  \
            --doc_column_names id,title,text  \
            --q_max_len $q_max_len  \
            --p_max_len $p_max_len  \
            --pos_token true  \
            --neg_token false  \
            --fp16  \
            --trec_run_path  /OpenMatch/experiments/inference.cocodr-base-msmarco.$task/test.trec \
            --trec_save_path /OpenMatch/results/beir/$name_prefix-$task.trec  \
            --dataloader_num_workers 1 \
            --reranking_depth 100
    elif [ ${task} == scifact ] || [ ${task} == trec-news ]
    then
        export q_max_len=64
        export p_max_len=256
        echo ${task}
        CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} -m openmatch.driver.rerank  \
            --output_dir None  \
            --model_name_or_path $chk\
            --per_device_eval_batch_size 256  \
            --query_path /dataset/beir/$task/queries.jsonl  \
            --corpus_path /dataset/beir/$task/corpus.jsonl  \
            --query_template "Query: <text>" \
            --doc_template "Document: <title> <text> Relevant: " \
            --query_column_names id,text  \
            --doc_column_names id,title,text  \
            --q_max_len $q_max_len  \
            --p_max_len $p_max_len  \
            --pos_token true  \
            --neg_token false  \
            --fp16  \
            --trec_run_path  /OpenMatch/experiments/inference.cocodr-base-msmarco.$task/test.trec \
            --trec_save_path /OpenMatch/results/beir/$name_prefix-$task.trec  \
            --dataloader_num_workers 1 \
            --reranking_depth 100
    elif [ ${task} == robust04 ]
    then
        export q_max_len=64
        export p_max_len=256
        echo ${task}
        CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} -m openmatch.driver.rerank  \
            --output_dir None  \
            --model_name_or_path $chk\
            --per_device_eval_batch_size 256  \
            --query_path /dataset/beir/$task/queries.jsonl  \
            --corpus_path /dataset/beir/$task/corpus.jsonl  \
            --query_template "Query: <text>" \
            --doc_template "Document: <text> Relevant: " \
            --query_column_names id,text  \
            --doc_column_names id,title,text  \
            --q_max_len $q_max_len  \
            --p_max_len $p_max_len  \
            --pos_token true  \
            --neg_token false  \
            --fp16  \
            --trec_run_path  /OpenMatch/experiments/inference.cocodr-base-msmarco.$task/test.trec \
            --trec_save_path /OpenMatch/results/beir/$name_prefix-$task.trec  \
            --dataloader_num_workers 1 \
            --reranking_depth 100
    else
        export q_max_len=64
        export p_max_len=128
        echo ${task}
        CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} -m openmatch.driver.rerank  \
            --output_dir None  \
            --model_name_or_path $chk\
            --per_device_eval_batch_size 256  \
            --query_path /dataset/beir/$task/queries.jsonl  \
            --corpus_path /dataset/beir/$task/corpus.jsonl  \
            --query_template "Query: <text>" \
            --doc_template "Document: <title> <text> Relevant: " \
            --query_column_names id,text  \
            --doc_column_names id,title,text  \
            --q_max_len $q_max_len  \
            --p_max_len $p_max_len  \
            --pos_token true  \
            --neg_token false  \
            --fp16  \
            --trec_run_path  /OpenMatch/experiments/inference.cocodr-base-msmarco.$task/test.trec \
            --trec_save_path /OpenMatch/results/beir/$name_prefix-$task.trec  \
            --dataloader_num_workers 1 \
            --reranking_depth 100\

    fi
done

done

