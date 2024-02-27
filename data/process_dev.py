import csv
import sys
import json
import numpy as np
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

trec='/dataset/msmarco/passage/dev_rank.tsv'
queries_path='/dataset/msmarco/passage/queries.dev.small.tsv'
docs_path='/dataset/msmarco/passage/collection.tsv'
qrels_path='/dataset/msmarco/passage/qrels.dev.small.tsv'
title_path='/dataset/msmarco/passage/corpus.tsv'


def query_process(query_path):
    queries = open(query_path)
    queries_read = csv.reader(queries, delimiter="\t")

    query_dict = {}
    for qid, query in queries_read:
        query_dict[qid] = query

    print('query process done')
    return query_dict

def doc_process(doc_path):
    docs = open(doc_path)
    docs_read = csv.reader(docs, delimiter="\t")

    doc_dict = {}
    for did, doc in docs_read:
        doc_dict[did] = doc

    print('doc process done')
    return doc_dict

def title_process(doc_path):
    docs = open(doc_path)
    docs_read = csv.reader(docs, delimiter="\t")

    title_dict = {}
    for did, title, doc in docs_read:
        title_dict[did] = title

    print('title process done')
    return title_dict

def qrel_process(qrel_path):
    qrels = open(qrel_path)
    qrels_read = csv.reader(qrels, delimiter="\t")

    qrel_dict = {}
    for qid, a, did, b in qrels_read:
        qrel_dict[qid] = did

    print('qrel process done')
    return qrel_dict

def process_rest(data, t, outfile):
    for i in range(100 - len(data['pos']) - len(data['neg'])):
        temp['doc'] = "None"
        temp['did'] = "-1"
        temp['label'] = 0
        temp['score'] = "-10"
        temp['rank'] = "100"
        data['neg'].append(temp)
    json.dump(data, outfile)
    outfile.write("\n")
    #print(count)
    t += 1  
    return t

def normalize_data(data, use_global=True):
    if len(data['pos']) > 0:
        score_max = float(data['pos'][0]['score'])
        score_min = float(data['pos'][0]['score'])
    else:
        score_max = float(data['neg'][0]['score'])
        score_min = float(data['neg'][0]['score'])
    for i in data['neg']:
        score_max = max(score_max, float(i['score']))
        score_min = min(score_min, float(i['score']))
    if use_global == True:
        score_min = 165
        score_max = 190
        diff = score_max - score_min
        if len(data['pos']) > 0:
            data['pos'][0]['score'] = str((min(score_max, float(data['pos'][0]['score'])) - score_min) / diff)
        for i in range(len(data['neg'])):
            data['neg'][i]['score'] = str((min(score_max, float(data['neg'][i]['score'])) - score_min) / diff)
    else:
        diff = score_max - score_min
        if len(data['pos']) > 0:
            data['pos'][0]['score'] = str((float(data['pos'][0]['score']) - score_min) / diff)
        for i in range(len(data['neg'])):
            data['neg'][i]['score'] = str((float(data['neg'][i]['score']) - score_min) / diff)
    return data

queries_dict = query_process(queries_path)
docs_dict = doc_process(docs_path)
qrels_dict = qrel_process(qrels_path)
title_dict = title_process(title_path)

count = 0
m = 100
t = 0
store = True
dev_num = 0
for k, v in queries_dict.items():
    dev_num += 1
print(dev_num)

rank = 0
with open('/dataset/msmarco/passage/dev_cocondenser_mine_global.json', 'w') as outfile:
    total_len = 0
    with open(trec, 'r') as f_run:
        total_len = len(f_run.readlines())
    data = {}
    with open(trec, 'r') as f_run:
        passs = "xxx"
        for i, line in tqdm(enumerate(f_run), total=total_len):

            #qid, _, did, rank, score, _= line.strip().split()
            qid, did, score = line.strip().split()
            #rank = int(rank)
            if passs == qid:
                continue
            if qid not in queries_dict or qid not in qrels_dict:
                continue
            if 'qid' not in data or ('qid' in data and data['qid'] != qid):
                if 'qid' in data:
                    t = process_rest(data, t, outfile)
                data = {}
                rank = 0
                query = queries_dict[qid]
                data['qid'] = qid
                data['query'] = query
                data['pos'] = []
                data['neg'] = []

                did_true = qrels_dict[qid]

                if did_true not in docs_dict.keys():
                    passs = qid
                    data = {}
                    rank += 1
                    count += 1
                    continue
            count += 1
            rank += 1

            if did in docs_dict and qrels_dict[qid] != did:
                temp = {}
                doc = docs_dict[did]
                doc = doc.replace('[SEP]',  '')
                doc = doc.replace('  ',  ' ')
                temp['doc'] = doc[:512]
                temp['did'] = did
                temp['label'] = 0
                temp['score'] = score
                temp['rank'] = str(rank)
                temp['title'] = title_dict[did]
                data['neg'].append(temp)

            if did in docs_dict and qrels_dict[qid] == did:
        
                temp = {}
                doc = docs_dict[did]
                doc = doc.replace('[SEP]',  '')
                doc = doc.replace('  ',  ' ')
                temp['doc'] = doc[:512]
                temp['did'] = did
                temp['label'] = 1
                temp['score'] = score
                temp['rank'] = str(rank)
                temp['title'] = title_dict[did]
                data['pos'].append(temp)

            if did not in docs_dict:
                temp['doc'] = "None"
                temp['did'] = "-1"
                temp['label'] = 0
                temp['score'] = "-10"
                temp['rank'] = "100"
                temp['title'] = "None"
                print("no")
            
            if len(data['neg']) + len(data['pos']) >= m:
                data = normalize_data(data)
                json.dump(data, outfile)
                #print(count)
                t += 1
                #print(t)
                outfile.write("\n")
                passs = qid
                data = {}
                rank = 0
        if 'qid' in data:
            t = process_rest(data, t, outfile)
        print(count)
        print(t)


with open('/dataset/msmarco/passage/dev_cocondenser_mine_global.trec', 'w') as w:
    with open('/dataset/msmarco/passage/dev_cocondenser_mine_global.json', 'r') as r:

        for i, line in enumerate(r):

            line = json.loads(line)
            if len(line['pos']) >= 1:
                for j in range(len(line['pos'])):
                    w.write(line['qid'] + ' 0 ' + line['pos'][j]['did'] + ' ' + str(line['pos'][j]['label']) + '\n')

            if len(line['pos']) == 0:
        
                w.write(line['qid'] + ' 0 ' + 'D' + ' ' + '1' + '\n')
