from typing import Dict, Any
import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import T5Tokenizer

import json

class t5Dataset(Dataset):
    def __init__(
        self,
        dataset: str,
        tokenizer: T5Tokenizer,
        max_query_len: int = 64,
        max_seq_len: int = 512,
        is_test: bool = False,
        doc_size: int = 100,
        max_input = 1280000,
        relieve_CLS=False,
    ) -> None:
        self._label_mapping=['false','true']
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_input = max_input
        self._template = "[CLS] Query: <q> Title: <t> Passage: <d> Relevant: " #" Query: <q> Document: <d> "
        self._max_query_len = max_query_len
        self._max_seq_len = max_seq_len
        self._is_test = is_test
        self._doc_size = doc_size
        self.relieve_CLS = relieve_CLS
        
        with open(self._dataset,'r') as f:
            self._examples = []
            for i, line in enumerate(f):
                line = json.loads(line)
                self._examples.append(line)        


    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]

        input_id_list = []
        attention_mask_list = []
        raw_label_list = []
        doc_id_list = []
        text_list = []
        tokenized_label_list = []
        score_token_ids = []

        doc_list = example['pos'] + example['neg']
        neg_examples = doc_list[1:]
        np.random.shuffle(neg_examples)
        doc_list[1:] = neg_examples

        grad_examples = doc_list[:self._doc_size]
        np.random.shuffle(grad_examples)
        doc_list[:self._doc_size] = grad_examples
        rating = []
        for i, line in enumerate(doc_list):
            text = self._template.replace("<q>", example["query"][:self._max_query_len]).replace("<d>", str(line['doc'])).replace("<t>", str(line['title']))
            tokenized = self._tokenizer(text, padding="max_length", truncation=True, max_length=self._max_seq_len) #384
            source_ids, source_mask = tokenized["input_ids"], tokenized["attention_mask"]
            tokenized_target = self._tokenizer(self._label_mapping[line['label']], padding="max_length", truncation=True, max_length=2)["input_ids"]
            tokenized_target = [
                (label if label != self._tokenizer.pad_token_id else -100) for label in tokenized_target
            ]
            text_list.append(text)
            input_id_list.append(source_ids)
            attention_mask_list.append(source_mask)
            doc_id_list.append(line['did'])
            raw_label_list.append(line['label'])
            tokenized_label_list.append(tokenized_target[0])
            score_token_id = 0 #CLS
            score_token_ids.append(score_token_id)
            if 'rating' in line:
                rating.append(line['rating'])
            #print(tokenized_target[0])
        

        output = {
            "texts": text_list,
            "input_ids": input_id_list,
            "attention_mask": attention_mask_list,
            'query_ids': [example['qid']] * len(doc_id_list),
            'doc_ids': doc_id_list,
            'score_token_ids': score_token_ids,
            'rating': rating,
        }
        if self._is_test:
            return output
        else:
            output['labels'] = tokenized_label_list
            output['raw_labels'] = raw_label_list
            return output

    def __len__(self) -> int:
        return len(self._examples)

    def collate(self, batch: Dict[str, Any]):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])

        texts = [item['texts'] for item in batch]
        query_ids = [item['query_ids'] for item in batch]
        doc_ids = [item['doc_ids'] for item in batch]
        raw_labels = [item['raw_labels'] for item in batch]
        rating = torch.tensor([item['rating'] for item in batch], dtype=torch.long)
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        score_token_ids = torch.tensor([item['score_token_ids'] for item in batch], dtype=torch.long)
        output = {
            "texts": texts,
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'query_ids': query_ids,
            'doc_ids': doc_ids,
            'score_token_ids': score_token_ids,
            'rating': rating,
        }

        if self._is_test:
            return output
        else:
            output['labels'] = labels
            output['raw_labels'] = raw_labels
            return output


class t5Dataset_score(Dataset):
    def __init__(
        self,
        dataset: str,
        tokenizer: T5Tokenizer,
        max_query_len: int = 64,
        max_seq_len: int = 512,
        is_test: bool = False,
        doc_size: int = 100,
        max_input = 1280000,
        bin_tokens=None,
        add_rank=False,
        relieve_CLS=False,
        number_bin=False,
    ) -> None:
        self._label_mapping=['false','true']
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_input = max_input
        self._template = "[CLS] Query: <q> Title: <t> score: <s> Passage: <d> Relevant: " #" Query: <q> Document: <d> "
        self._max_query_len = max_query_len
        self._max_seq_len = max_seq_len 
        self._is_test = is_test
        self._doc_size = 100 #doc_size
        self._bin_tokens = bin_tokens
        self._bin_tokens_ids = []
        self.add_rank = add_rank
        self.relieve_CLS = relieve_CLS
        self.number_bin = number_bin

        for token in bin_tokens:
            self._bin_tokens_ids.append(tokenizer(token, max_length=4, truncation=True)["input_ids"][0])

        with open(self._dataset,'r') as f:
            self._examples = []
            for i, line in enumerate(f):
                line = json.loads(line)
                self._examples.append(line)        


    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]

        input_id_list = []
        attention_mask_list = []
        raw_label_list = []
        doc_id_list = []
        text_list = []
        tokenized_label_list = []
        score_token_ids = []
        score_memory = []

        doc_list = example['pos'] + example['neg']
        neg_examples = doc_list[1:]
        np.random.shuffle(neg_examples)
        doc_list[1:] = neg_examples

        grad_examples = doc_list[:self._doc_size]
        np.random.shuffle(grad_examples)
        doc_list[:self._doc_size] = grad_examples
        
        rating = []
        for i, line in enumerate(doc_list):
            if self._bin_tokens is not None:
                if self.add_rank:
                    bin_n = int(min(100, float(line['rank'])) / 10)
                    now_bin_token = self._bin_tokens[bin_n]
                elif self.number_bin:
                    bin_n = int(float(line['score']) * 100) #100
                    now_bin_token = str(bin_n)
                else:
                    #bin_n = int(float(line['score']) * 10) #10
                    bin_n = int(float(line['score']) * 100) #100
                    #bin_n = int(float(line['score']) * 20) #20
                    #bin_n = int(float(line['score']) * 5) #5
                    now_bin_token = self._bin_tokens[bin_n]             
                text = self._template.replace("<q>", example["query"][:self._max_query_len]) \
                    .replace("<s>", now_bin_token).replace("<d>", str(line['doc'])).replace("<t>", str(line["title"]))
                #text = self._template.replace("<q>", example["query"][:self._max_query_len]) \
                #    .replace("score: <s>", "").replace("<d>", str(line['doc'])).replace("<t>", str(line["title"]))
            else:
                if self.add_rank:
                    score = round(float(line['rank']) / 100, 2)
                else:
                    score = round(float(line['score']), 2)
                text = self._template.replace("<q>", example["query"][:self._max_query_len]) \
                    .replace("<s>", str(score)).replace("<d>", str(line['doc'])).replace("<t>", str(line["title"]))
            tokenized = self._tokenizer(text, padding="max_length", truncation=True, max_length=self._max_seq_len) #384
            source_ids, source_mask = tokenized["input_ids"], tokenized["attention_mask"]
            tokenized_target = self._tokenizer(self._label_mapping[line['label']], padding="max_length", truncation=True, max_length=2)["input_ids"]
            tokenized_target = [
                (label if label != self._tokenizer.pad_token_id else -100) for label in tokenized_target
            ]
            # score_token_id = 0 #[CLS]
            # for i, token in enumerate(source_ids):
            #     if token in self._bin_tokens_ids:
            #         score_token_id = i
            # if self.relieve_CLS:
            score_token_id = 0 #CLS
            score_token_ids.append(score_token_id)
            score_memory.append(bin_n)
            text_list.append(text)
            input_id_list.append(source_ids)
            attention_mask_list.append(source_mask)
            doc_id_list.append(line['did'])
            raw_label_list.append(line['label'])
            if 'rating' in line:
                rating.append(line['rating'])
            tokenized_label_list.append(tokenized_target[0])

        output = {
            "texts": text_list,
            "input_ids": input_id_list,
            "attention_mask": attention_mask_list,
            'query_ids': [example['qid']] * len(doc_id_list),
            'doc_ids': doc_id_list,
            'score_token_ids': score_token_ids,
            'score_memory': score_memory,
            'rating': rating,
        }
        if self._is_test:
            return output
        else:
            output['labels'] = tokenized_label_list
            output['raw_labels'] = raw_label_list
            return output

    def __len__(self) -> int:
        return len(self._examples)

    def collate(self, batch: Dict[str, Any]):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])

        texts = [item['texts'] for item in batch]
        query_ids = [item['query_ids'] for item in batch]
        doc_ids = [item['doc_ids'] for item in batch]
        raw_labels = [item['raw_labels'] for item in batch]
        rating = torch.tensor([item['rating'] for item in batch], dtype=torch.long)
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        score_token_ids = torch.tensor([item['score_token_ids'] for item in batch], dtype=torch.long)
        score_memory = torch.tensor([item['score_memory'] for item in batch], dtype=torch.long)
        output = {
            "texts": texts,
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'query_ids': query_ids,
            'doc_ids': doc_ids,
            'score_token_ids': score_token_ids,
            'score_memory': score_memory,
            "rating": rating,
        }

        if self._is_test:
            return output
        else:
            output['labels'] = labels
            output['raw_labels'] = raw_labels
            return output