import csv
from curses import use_default_colors
import sys
import json
import numpy as np
from tqdm import tqdm
import os
import random

csv.field_size_limit(sys.maxsize)

trec="/msmarco/coCondenser/rank.tsv"
queries_path='/msmarco/queries.train.tsv'
docs_path='/msmarco/collection.tsv'
qrels_path='/msmarco/qrels.train.tsv'
title_path='/collections/marco/corpus.tsv'


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
m = 1000
need_m = 100
need_random = False #True
score_normalize = True
t = 0
store = True
data  = {}
last_qid = "xxxx"
train_split_num = 495000
outfile_train = open('/msmarco/coCondenser/train_cocondenser_495000_global.json', 'w')
outfile_dev = open('/msmarco/coCondenser/new_train_cocondenser_dev_3195_global.json', 'w')
total_len = 0
with open(trec, 'r') as f_run:
    total_len = len(f_run.readlines())
data = {}
rank = 0
with open(trec, 'r') as f_run:
    passs = "xxx"
    for i, line in tqdm(enumerate(f_run), total=total_len):

        qid, did, score = line.strip().split()
        """if passs == qid:
            continue
        if qid not in queries_dict or qid not in qrels_dict:
            continue"""
        if rank == 0:
            """if len(data['pos']) < 1:
                temp = {}
                doc = docs_dict[qrels_dict[qid]]
                doc = doc.replace('[SEP]',  '')
                doc = doc.replace('  ',  ' ')
                temp['doc'] = doc[:512]
                temp['did'] = qrels_dict[qid]
                temp['label'] = 1
                temp['score'] = data['neg'][-1]['score']
                temp['rank'] = data['neg'][-1]['rank']
                data['pos'].append(temp)
            if need_random == True:
                random.shuffle(data['neg'])
            if len(data['neg']) >= need_m - 1:
                data['neg'] = data['neg'][:need_m - 1]
                json.dump(data, outfile)
                t += 1
                outfile.write("\n")"""
            passs = qid
            data = {}
            last_qid = qid
        if 'qid' not in data or ('qid' in data and data['qid'] != qid):

            data = {}
            query = queries_dict[qid]
            data['qid'] = qid
            data['query'] = query
            data['pos'] = []
            data['neg'] = []
            rank = 0

            did_true = qrels_dict[qid]

            if did_true not in docs_dict.keys():
                passs = qid
                data = {}
                count += 1
                rank += 1
                continue
        last_qid = qid
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
            temp['score'] = str(float(score))
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
            temp['score'] = str(float(score))
            temp['rank'] = str(rank)
            temp['title'] = title_dict[did]
            data['pos'].append(temp)

        if did not in docs_dict:
            print("no")
            continue
        
        if rank == m:
            if len(data['pos']) >= 1:
                if t < train_split_num:
                    if need_random == True:
                        random.shuffle(data['neg'])
                    data['neg'] = data['neg'][:need_m - 1]
                    if score_normalize == True:
                        data = normalize_data(data)
                    json.dump(data, outfile_train)
                    #print(count)
                    
                    #print(t)
                    outfile_train.write("\n")
                else:
                    if need_random == True:
                        random.shuffle(data['neg'])
                    if int(data['pos'][0]['rank']) > 100:
                        data['pos'] = []
                        data['neg'] = data['neg'][:need_m]
                    else:
                        data['neg'] = data['neg'][:need_m - 1]
                    if score_normalize == True:
                        data = normalize_data(data)
                    json.dump(data, outfile_dev)
                    #print(count)
                    
                    #print(t)
                    outfile_dev.write("\n")
                t += 1
            else:
                if t >= train_split_num:
                    if need_random == True:
                        random.shuffle(data['neg'])
                    data['pos'] = []
                    data['neg'] = data['neg'][:need_m]
                    if score_normalize == True:
                        data = normalize_data(data)
                    json.dump(data, outfile_dev)
                    t += 1
            passs = qid
            data = {}
            last_qid = qid
            rank = 0
    print(count) # 66287163 #condenser: 59527549
    print(t) #662830 #condenser: 498195   condenser random:502939

outfile_dev.close()
with open('/msmarco/coCondenser/train_cocondenser_dev_3195_global.trec', 'w') as w:
    with open('/msmarco/coCondenser/train_cocondenser_dev_3195_global.json', 'r') as r:

        for i, line in enumerate(r):

            line = json.loads(line)
            if len(line['pos']) == 1:

                w.write(line['qid'] + ' 0 ' + line['pos'][0]['did'] + ' ' + str(line['pos'][0]['label']) + '\n')

            if len(line['pos']) == 0:
        
                w.write(line['qid'] + ' 0 ' + 'D' + ' ' + '1' + '\n')