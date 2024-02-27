# OpenMatch
## Train
For bert-like models training
```shell
sh train_bert.sh
```

For edrm, cknrm, knrm or tk training
```shell
sh train.sh
```

### Options
```
-task             'ranking': pair-wise, 'classification': query-doc.
-model            'bert', 'tk', 'edrm', 'cknrm' or 'knrm'.
-reinfoselect     use reinfoselect or not.
-resset           reset the model or not, used in reinfoselect setting.
-train            path to training dataset.
-max_input        max input of instances.
-save             path for saving model checkpoint.
-dev              path to dev dataset.
-qrels            path to qrels.
-vocab            path to glove or customized vocab.
-ent_vocab        path to entity vocab, for edrm.
-pretrain         path to pretrained bert model.
-checkpoint       path to checkpoint.
-res              path for saving result.
-metric           which metrics to use, e.g. ndcg_cut_10.
-mode             use cls or pooling as bert representation.
-n_kernels        kernel number, for tk, edrm, cknrm or knrm.
-max_query_len    max length of query tokens.
-max_doc_len      max length of document tokens.
-maxp             bert max passage.
-epoch            how many epoch.
-batch_size       batch size.
-lr               learning rate.
-n_warmup_steps   warmup steps.
-eval_every       e.g. 1000, every 1000 steps evaluate on dev data.
```

## Inference
For bert-like models inference
```shell
sh inference_bert.sh
```

For edrm, cknrm, knrm or tk inference
```shell
sh inference.sh
```

### Options
```
-task             'ranking': pair-wise, 'classification': query-doc.
-model            'bert', 'tk', 'edrm', 'cknrm' or 'knrm'.
-max_input        max input of instances.
-test             path to test dataset.
-vocab            path to glove or customized vocab.
-ent_vocab        path to entity vocab.
-pretrain         path to pretrained bert model.
-checkpoint       path to checkpoint.
-res              path for saving result.
-n_kernels        kernel number, for tk, edrm, cknrm or knrm.
-max_query_len    max length of query tokens.
-max_doc_len      max length of document tokens.
-batch_size       batch size.
```

## Data Format
### Ranking Task
For bert, tk, cknrm or knrm:

|file|format|
|:---|:-----|
|train|{"query": str, "doc\_pos": str, "doc\_neg": str}|
|dev  |{"query": str, "doc": str, "label": int, "query\_id": str, "doc\_id": str, "retrieval\_score": float}|
|test |{"query": str, "doc": str, "query\_id": str, "doc\_id": str, "retrieval\_score": float}|

For edrm:

|file|format|
|:---|:-----|
|train|+{"query\_ent": list, "doc\_pos\_ent": list, "doc\_neg\_ent": list, "query\_des": list, "doc\_pos\_des": list, "doc\_neg\_des": list}|
|dev  |+{"query\_ent": list, "doc\_ent": list, "query\_des": list, "doc\_des": list}|
|test |+{"query\_ent": list, "doc\_ent": list, "query\_des": list, "doc\_des": list}|

The *query_ent*, *doc_ent* is a list of entities relevant to the query or document, *query_des* is a list of entity descriptions.

### Classification Task
Only train file format different with ranking task.

For bert, tk, cknrm or knrm:

|file|format|
|:---|:-----|
|train|{"query": str, "doc": str, "label": int}|

For edrm:

|file|format|
|:---|:-----|
|train|+{"query\_ent": list, "doc\_ent": list, "query\_des": list, "doc\_des": list}|

### Others
The dev and test files can be set as:
```
-dev queries={path to queries},docs={path to docs},qrels={path to qrels},trec={path to trec}
-test queries={path to queries},docs={path to docs},trec={path to trec}
```

|file|format|
|:---|:-----|
|queries|{"query\_id":, "query":}|
|docs|{"doc\_id":, "doc":}|
|qrels|query\_id iteration doc\_id label|
|trec|query\_id Q0 doc\_id rank score run-tag|

For edrm, the queries and docs are a little different:

|file|format|
|:---|:-----|
|queries|+{"query\_ent": list, "query\_des": list}|
|docs|+{"doc\_ent": list, "doc\_des": list}|

Other bert-like models are also available, e.g. electra, scibert. You just need to change the path to the vocab and the pretrained model.

You can also train bert for masked language model with *train_bertmlm.py*. The train file format is as follows:

|file|format|
|:---|:-----|
|train|{'doc': str}|

If you want to concatenate the neural features with retrieval scores (SDM/BM25), and run coor-ascent, you need to generate a features file using *gen_feature.py*, and run
```
sh coor_ascent.sh
```
