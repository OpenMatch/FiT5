from typing import List, Tuple, Dict, Any
import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer

import json

class bertDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        tokenizer: BertTokenizer,
        max_query_len: int = 64,
        max_seq_len: int = 512,
        is_test: bool = False,
        max_input = 1280000
    ) -> None:
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_input = max_input
        self._max_query_len = max_query_len
        self._max_seq_len = max_seq_len
        self._template = " Query: <q> Document: <d> "
        self._is_test = is_test

        with open(self._dataset,'r') as f:
            self._examples = []
            for i, line in enumerate(f):
                line = json.loads(line)
                self._examples.append(line)

    # def pack_bert_features(self, query_tokens: List[str], doc_tokens: List[str]):
    #     input_tokens = [self._tokenizer.cls_token] + query_tokens + [self._tokenizer.sep_token] + doc_tokens + [self._tokenizer.sep_token]
    #     input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
    #     segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_tokens) + 1)
    #     input_mask = [1] * len(input_tokens)

    #     padding_len = self._max_seq_len - len(input_ids)
    #     input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
    #     input_mask = input_mask + [0] * padding_len
    #     segment_ids = segment_ids + [0] * padding_len

    #     assert len(input_ids) == self._max_seq_len
    #     assert len(input_mask) == self._max_seq_len
    #     assert len(segment_ids) == self._max_seq_len

    #     return input_ids, input_mask, segment_ids   


    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]

        #query_tokens = self._tokenizer.tokenize(example['query'])[:self._max_query_len]

        query_text_list = []
        doc_text_list = []
        input_id_list = []
        input_mask_list = []
        segment_id_list = []
        label_list = []
        doc_id_list = []

        doc_list = example['pos'] + example['neg']
        neg_examples = doc_list[1:]
        np.random.shuffle(neg_examples)
        doc_list[1:] = neg_examples

        for i, line in enumerate(doc_list):

            query_text = str(example["query"][:self._max_query_len])
            doc_text = str((line['doc'])[:self._max_seq_len-len(query_text)-3])
            #text = self._template.replace("<q>", example["query"][:self._max_query_len]).replace("<d>", str(line['doc']))[:self._max_seq_len]
            #doc_tokens = self._tokenizer.tokenize(line['doc'])[:self._max_seq_len-len(query_tokens)-3]
            #tokenized = self._tokenizer.encode_plus(text, padding="max_length", truncation=True, max_length=384)
            tokenized = self._tokenizer.encode_plus(text = query_text, text_pair = doc_text, padding="max_length", truncation=True, max_length=384)
            input_ids, input_mask, segment_ids = tokenized["input_ids"], tokenized["attention_mask"],tokenized["token_type_ids"]

            query_text_list.append(query_text)
            doc_text_list.append(doc_text)
            input_id_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_id_list.append(segment_ids)
            doc_id_list.append(line['did'])
            label_list.append(line['label'])

        output = {
            "query_texts": query_text_list,
            "doc_texts": doc_text_list,
            "input_ids": input_id_list,
            "input_mask": input_mask_list,
            "segment_ids": segment_id_list,
            'query_ids': [example['qid']] * len(doc_id_list),
            'doc_ids': doc_id_list,
        }
        if self._is_test:
            return output
        else:
            output['labels'] = label_list
            return output

    def __len__(self) -> int:
        return len(self._examples)

    def collate(self, batch: Dict[str, Any]):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        input_mask = torch.tensor([item['input_mask'] for item in batch])
        segment_ids = torch.tensor([item['segment_ids'] for item in batch])

        query_texts = [item['query_texts'] for item in batch]
        doc_texts = [item['doc_texts'] for item in batch]
        query_ids = [item['query_ids'] for item in batch]
        doc_ids = [item['doc_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        output = {
            'query_texts': query_texts,
            'doc_texts': doc_texts,
            'input_ids': input_ids, 
            'input_mask': input_mask,
            "segment_ids": segment_ids,
            'query_ids': query_ids,
            'doc_ids': doc_ids,
        }

        if self._is_test:
            return output
        else:
            output['labels'] = labels
            return output