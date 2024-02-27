import os
from argparse import Action

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler, Dataset, TensorDataset, IterableDataset
torch.multiprocessing.set_sharing_strategy('file_system')
import glob
from collections import OrderedDict

import random
import numpy as np
import logging
import pytrec_eval


def set_dist_args(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(
                args.server_ip,
                args.server_port),
            redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        args.n_gpu = 1
    args.device = device
   
    # Set seed
    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # store args for multi process
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()
    # assign args.world_size
    else:
        args.world_size = 1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logging.warning(
        # "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        "Process gpu rank: %s, process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        args.rank if (args.local_rank != -1) else 0,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        # args.fp16,
    )

def merge_resfile(split_pattern, output_file):
    splits = glob.glob(split_pattern)
    logging.info("temperary validation trec files: {}".format(splits))
    res_dict = {}
    for s in splits:
        with open(s,'r') as f:
            for line in f:
                qid, _, pid, _, score, _ = line.strip().split() # ranking is meaningless in distributed inference
                score = float(score)
                if qid not in res_dict:
                    res_dict[qid]=[(pid,score)]
                else:
                    res_dict[qid].append((pid,score))
        os.remove(s)
    cnt = 0
    with open(output_file,'w') as f:
        for qid in res_dict:
            res_dict[qid] = sorted(res_dict[qid], key=lambda x: x[1], reverse=True)
            rank = 1 # start from 1
            for pid, score in res_dict[qid]:
                f.write(qid+' Q0 '+ str(pid) +' '+str(rank)+' '+ str(score) +' FiT5\n')
                rank+=1
                cnt+=1
    logging.info("merge total {} lines".format(cnt))

# https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py
class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default
    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.
    Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    Example::
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch

class DictOrStr(Action):
    def __call__(self, parser, namespace, values, option_string=None):
         if '=' in values:
             my_dict = {}
             for kv in values.split(","):
                 k,v = kv.split("=")
                 my_dict[k] = v
             setattr(namespace, self.dest, my_dict)
         else:
             setattr(namespace, self.dest, values)

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_trec(rst_file, rst_dict):
    with open(rst_file, 'w') as writer:
        for q_id, scores in rst_dict.items():
            res = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
            for rank, value in enumerate(res):
                writer.write(str(q_id) + ' Q0 ' + str(value[0]) + ' ' + str(rank+1) + ' ' + str(value[1][0]) + ' openmatch\n')
    return

def save_features(rst_file, features):
    with open(rst_file, 'w') as writer:
        for feature in features:
            writer.write(feature+'\n')
    return

def init_logger(args):

    device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
   
    # Set seed
    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
def get_ndcg(qrels: str, trec: str, metric: str = 'ndcg_cut_10') -> float:
    k = int(metric.split('_')[-1])

    qrel = {}
    with open(qrels, 'r') as f_qrel:
        for line in f_qrel:
            line_split = line.strip().split()
            if len(line_split) == 4:
                qid, _, did, label = line_split
            else:
                qid, did, label = line_split
            if qid == 'query-id': continue
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][did] = int(label)

    run = {}
    with open(trec, 'r') as f_run:
        for line in f_run:
            qid, _, did, _, score, _ = line.strip().split()
            if qid not in run:
                run[qid] = {}
            run[qid][did] = float(score)
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {'ndcg_cut.10'})
    eval_results = evaluator.evaluate(run) 
    ndcg10 = 0
    num = 0
    for k, v in eval_results.items():
        ndcg10 += v['ndcg_cut_10']
        num += 1
    return ndcg10 / num
def get_mrr(qrels: str, trec: str, metric: str = 'mrr_cut_10') -> float:
    k = int(metric.split('_')[-1])

    qrel = {}
    with open(qrels, 'r') as f_qrel:
        for line in f_qrel:
            line_split = line.strip().split()
            if len(line_split) == 4:
                qid, _, did, label = line_split
            else:
                qid, did, label = line_split
            if qid == 'query-id': continue
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][did] = int(label)

    run = {}
    with open(trec, 'r') as f_run:
        for line in f_run:
            qid, _, did, _, _, _ = line.strip().split()
            if qid not in run:
                run[qid] = []
            run[qid].append(did)
    
    mrr = 0.0
    for qid in run:
        rr = 0.0
        for i, did in enumerate(run[qid][:k]):
            if qid in qrel and did in qrel[qid] and qrel[qid][did] > 0:
                rr = 1 / (i+1)
                break
        mrr += rr
    mrr /= len(run)
    return mrr

def get_mrr_rev(qrels: str, trec: str, metric: str = 'mrr_cut_10') -> float:
    k = int(metric.split('_')[-1])

    qrel = {}
    with open(qrels, 'r') as f_qrel:
        for line in f_qrel:
            qid, _, did, label = line.strip().split()
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][did] = int(label)

    run = {}
    with open(trec, 'r') as f_run:
        for line in f_run:
            qid, _, did, _, _, _ = line.strip().split()
            if qid not in run:
                run[qid] = []
            run[qid].append(did)
    
    mrr = 0.0
    for qid in run:
        rr = 0.0
        for i, did in enumerate(run[qid][:k]):
            if qid in qrel and did in qrel[qid] and qrel[qid][did] > 0:
                tot_length = len(run[qid][:k])
                rr = 1 / (tot_length-i+1)
                break
        mrr += rr
    mrr /= len(run)
    return mrr

# create new OrderedDict that does not contain `module.`
def clean_dict_name(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


class ListwiseLoss(nn.Module):


    def __init__(self) -> None:
        super(ListwiseLoss, self).__init__()

    def forward(self, input, target):
        """
        :param input: tensor (doc size, [false, true])
        :param target: tensor (doc size)
        """
        score_list = input[:, 1]
        pos_w = torch.argmax(target)
        log_softmax = nn.LogSoftmax()(score_list)
        return -log_softmax[pos_w]
# def get_mrr_test(qrels: str, trec: str, metric: str = 'mrr_cut_10') -> float:
#     k = int(metric.split('_')[-1])

#     qrel = {}
#     with open(qrels, 'r') as f_qrel:
#         for line in f_qrel:
#             qid, _, did, label = line.strip().split()
#             if qid not in qrel:
#                 qrel[qid] = {}
#             qrel[qid][did] = int(label)

#     run = {}
#     with open(trec, 'r') as f_run:
#         for line in f_run:
#             qid, _, did, _, _ = line.strip().split()
#             if qid not in run:
#                 run[qid] = []
#             run[qid].append(did)
    
#     mrr = 0.0
#     for qid in run:
#         rr = 0.0
#         for i, did in enumerate(run[qid][:k]):
#             if qid in qrel and did in qrel[qid] and qrel[qid][did] > 0:
#                 rr = 1 / (i+1)
#                 break
#         mrr += rr
#     mrr /= len(run)
#     return mrr
if __name__ == "__main__":
    fn = ListwiseLoss()
    input = torch.tensor([[0.6, 0.4], [0.2, 0.8]])
    label = torch.tensor([0, 1])
    loss = fn(input, label)
    print(loss)