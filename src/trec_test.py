import json

def get_mrr(qrels: str, trec: str, qrelss, metric: str = 'mrr_cut_10') -> float:
    k = int(metric.split('_')[-1])

    qrel = {}
    with open(qrels, 'r') as f_qrel:
        for line in f_qrel:
            qid, _, did, label = line.strip().split()
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][did] = 1

    qrell = {}
    with open(qrelss, 'r') as f_qrel:
        for line in f_qrel:
            qid, _, did, label = line.strip().split()
            if qid not in qrell:
                qrell[qid] = {}
            qrell[qid][did] = 1

    run = {}
    with open(trec, 'r') as f_run:
        for line in f_run:
            qid, _, did, _, _, _ = line.strip().split()
            if qid not in run:
                run[qid] = []
            run[qid].append(did)
    
    mrr = 0.0
    mrrr = 0.0
    for qid in run:
        rr = 0.0
        rrr = 0.0
        if qid == '1048585':
            ans = 1
        for i, did in enumerate(run[qid][:k]):
            if qid in qrel and did in qrel[qid] and qrel[qid][did] > 0:
                rr = 1 / (i+1)
                #break
            if qid in qrell and did in qrell[qid] and qrell[qid][did] > 0:
                rrr = 1 / (i+1)
                break
        mrr += rr
        mrrr += rrr
        if mrr != mrrr:
            print(qid)
            print(qrel[qid])
            print(qrell[qid])
            print(mrr)
            print(mrrr)
            break
    mrr /= len(run)
    return mrr

if __name__ == '__main__':
    passage_trec = "/checkpoints_local/t5/v10_passage_global_monot5_detach_fp16_0828/v10_passage_global_monot5_detach_fp16_0828-130000-test-bm25.trec"
    document_trec = "/checkpoints_local/t5/v10_global_detach_fp16_0607/v10_global_detach_fp16_0607-80000-test.trec"
    passage_qrel = "/dataset/msmarco/passage/qrels.dev.small.tsv"
    passage_qrel_ours = "/dataset/msmarco/passage/dev_ys_full.trec"
    mrr10 = get_mrr(passage_qrel, passage_trec, passage_qrel_ours)
    print(mrr10)
    
