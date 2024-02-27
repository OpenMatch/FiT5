import csv
from curses import use_default_colors
import sys
import json
import numpy as np
from tqdm import tqdm
import os
import random
import jsonlines

csv.field_size_limit(sys.maxsize)

dir = "/OpenMatch/experiments"

def query_process(query_path):
    query_dict = {}
    with open(query_path, 'r') as f:
        for i in f:
            data = json.loads(i)
            query_dict[data['_id']] = data['text']
    print('query process done')
    return query_dict

def doc_process(doc_path):
    doc_dict = {}
    with open(doc_path, 'r') as f:
        for i in f:
            data = json.loads(i)
            doc_dict[data['_id']] = {'text': data['text'], 'title': data['title']}

    print('doc process done')
    return doc_dict

def qrel_process(qrel_path):
    qrels = open(qrel_path)
    qrels_read = csv.reader(qrels, delimiter="\t")

    qrel_dict = {}
    for qid, did, b in qrels_read:
        if qid not in qrel_dict:
            qrel_dict[qid] = {}
        if b == 'score': continue
        qrel_dict[qid][did] = int(b)

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
        score_min = 206
        score_max = 217
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

def process_rest(data, t, outfile):
    for i in range(100 - len(data['pos']) - len(data['neg'])):
        temp['doc'] = "None"
        temp['did'] = "-1"
        temp['label'] = 0
        temp['score'] = "0"
        temp['rank'] = "100"
        data['neg'].append(temp)
    json.dump(data, outfile)
    outfile.write("\n")
    #print(count)
    t += 1  
    return t

for file in os.listdir(dir):
    trec = os.path.join(dir, file, 'test.trec')
    if not os.path.exists(trec): continue
    
    task_name = file.split('.')[-1]
    if task_name == "msmarco-train": continue
    #if task_name != "msmarco": continue

    queries_path='/dataset/beir/{}/queries.jsonl'.format(task_name)
    docs_path='/dataset/beir/{}/corpus.jsonl'.format(task_name)
    qrels_path='/dataset/beir/{}/qrels/test.tsv'.format(task_name)

    if not os.path.exists(queries_path):
        queries_path='/dataset/beir/cqadupstack/{}/queries.jsonl'.format(task_name)
        docs_path='/dataset/beir/cqadupstack/{}/corpus.jsonl'.format(task_name)
        qrels_path='/dataset/beir/cqadupstack/{}/qrels/test.tsv'.format(task_name)


    queries_dict = query_process(queries_path)
    docs_dict = doc_process(docs_path)
    qrels_dict = qrel_process(qrels_path)

    count = 0
    m = 100
    t = 0
    store = True
    dev_num = 0
    for k, v in queries_dict.items():
        dev_num += 1
    print(dev_num)

    rank = 0
    out_file = os.path.join("/monoT5/dataset/monoT5_beir", "dev_{}_monoT5_global.jsonl".format(task_name))
    with jsonlines.open(out_file, 'w') as outfile:
        total_len = 0
        with open(trec, 'r') as f_run:
            total_len = len(f_run.readlines())
        data = {}
        with open(trec, 'r') as f_run:
            passs = "xxx"
            for i, line in tqdm(enumerate(f_run), total=total_len):

                qid, _, did, _, score, _= line.strip().split()
                #qid, did, score = line.strip().split()
                #rank = int(rank)
                
                query = queries_dict[qid]
                title = ""
                if docs_dict[did]['title'] is not None:
                    title = docs_dict[did]['title']
                doc = title + " " + docs_dict[did]['text']
                doc = doc.replace('[SEP]',  '')
                doc = doc.replace('  ',  ' ')
                outfile.write({'query': query, "doc": doc[:512], "label": 0, "query_id": qid, "doc_id": did})
                """if did in docs_dict and (did not in qrels_dict or qrels_dict[qid][did] == 0):
                    temp = {}
                    doc = docs_dict[did]['text']
                    doc = doc.replace('[SEP]',  '')
                    doc = doc.replace('  ',  ' ')
                    outfile.write({'query': query, "doc": doc[:512], "label": 0, "query_id": qid, "doc_id": did})

                if did in docs_dict and did in qrels_dict and qrels_dict[qid][did] == 1:
            
                    temp = {}
                    doc = docs_dict[did]['text']
                    doc = doc.replace('[SEP]',  '')
                    doc = doc.replace('  ',  ' ')
                    outfile.write({'query': query, "doc": doc[:512], "label": 1, "query_id": qid, "doc_id": did})"""