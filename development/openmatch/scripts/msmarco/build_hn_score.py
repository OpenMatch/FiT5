# Adapted from Tevatron (https://github.com/texttron/tevatron)

from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import random
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from openmatch.utils import SimpleTrainPreProcessor as TrainPreProcessor

def get_cocondenser_score(score):
    score = float(score)
    score_min = 165
    score_max = 190
    diff = score_max - score_min
    return str(int((min(score_max, score) - score_min) / diff * 100))
def load_ranking(rank_file, relevance, n_sample, depth):
    with open(rank_file) as rf:
        lines = iter(rf)
        #q_0, _, p_0, _, _, _ = next(lines).strip().split()
        q_0, p_0, score = next(lines).strip().split()

        curr_q = q_0
        have_score_relevance = {}
        for p in relevance[q_0]:
            have_score_relevance[p] = "-1" if p != p_0 else get_cocondenser_score(score)
        negatives = [] if p_0 in relevance[q_0] else [p_0]
        negatives_score = [] if p_0 in relevance[q_0] else [get_cocondenser_score(score)]
        while True:
            try:
                #q, _, p, _, _, _ = next(lines).strip().split()
                q, p, score = next(lines).strip().split()
                if q != curr_q:
                    negatives = negatives[:depth]
                    negatives_score = negatives_score[:depth]
                    indice = list(range(len(negatives)))
                    random.shuffle(indice)
                    negatives = [negatives[i] for i in indice]
                    negatives_score = [negatives_score[i] for i in indice]
                    new_pos_q = []
                    new_pos_score = []
                    for k, v in have_score_relevance.items():
                        if v != "-1":
                            new_pos_q.append(k)
                            new_pos_score.append(v)
                    if len(new_pos_q) > 0:
                        yield curr_q, new_pos_q, new_pos_score, negatives[:n_sample], negatives_score[:n_sample]
                    curr_q = q
                    have_score_relevance = {}
                    for p in relevance[q_0]:
                        have_score_relevance[p] = "-1" if p != p_0 else get_cocondenser_score(score)
                    negatives = [] if p in relevance[q] else [p]
                    negatives_score = [] if p_0 in relevance[q_0] else [get_cocondenser_score(score)]
                else:
                    if p not in relevance[q]:
                        negatives.append(p)
                        negatives_score.append(get_cocondenser_score(score))
                    else:
                        have_score_relevance[p] = get_cocondenser_score(score)
            except StopIteration:
                negatives = negatives[:depth]
                negatives_score = negatives_score[:depth]
                indice = list(range(len(negatives)))
                random.shuffle(indice)
                negatives = [negatives[i] for i in indice]
                negatives_score = [negatives_score[i] for i in indice]
                new_pos_q = []
                new_pos_score = []
                for k, v in have_score_relevance.items():
                    if v != "-1":
                        new_pos_q.append(k)
                        new_pos_score.append(v)
                if len(new_pos_q) > 0:
                    yield curr_q, new_pos_q, new_pos_score, negatives[:n_sample], negatives_score[:n_sample]
                return


random.seed(datetime.now())
parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--hn_file', required=True)
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--doc_template', type=str, default=None)
parser.add_argument('--query_template', type=str, default=None)

parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--n_sample', type=int, default=30)
parser.add_argument('--depth', type=int, default=200)
parser.add_argument('--mp_chunk_size', type=int, default=500)
parser.add_argument('--shard_size', type=int, default=45000)

args = parser.parse_args()

qrel = TrainPreProcessor.read_qrel(args.qrels)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = TrainPreProcessor(
    query_file=args.queries,
    collection_file=args.collection,
    tokenizer=tokenizer,
    doc_max_len=args.truncate,
    doc_template=args.doc_template,
    query_template=args.query_template,
    allow_not_found=True
)

counter = 0
shard_id = 0
f = None
os.makedirs(args.save_to, exist_ok=True)

pbar = tqdm(load_ranking(args.hn_file, qrel, args.n_sample, args.depth))
with Pool() as p:
    for x in p.imap(processor.process_one_have_score, pbar, chunksize=args.mp_chunk_size):
        counter += 1
        if f is None:
            f = open(os.path.join(args.save_to, f'split{shard_id:02d}.hn.jsonl'), 'w')
            pbar.set_description(f'split - {shard_id:02d}')
        f.write(x + '\n')

        if counter == args.shard_size:
            f.close()
            f = None
            shard_id += 1
            counter = 0

if f is not None:
    f.close()