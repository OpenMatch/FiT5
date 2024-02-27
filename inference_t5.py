import sys
import argparse

import torch

from transformers import T5Tokenizer

import src as om
from src.utils import init_logger, save_trec, get_mrr, clean_dict_name, merge_resfile, get_ndcg, DistributedEvalSampler, set_dist_args

import logging
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
import os
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test(args, model, test_loader):
    rst_dict = {}
    for test_batch in tqdm(test_loader, disable=args.local_rank not in [-1, 0]):
        query_ids, doc_ids, labels = test_batch['query_ids'], test_batch['doc_ids'], test_batch['labels']

        input_id_list = test_batch['input_ids'].to(args.device) # bs * 100 * 384
        attention_mask_list = test_batch['attention_mask'].to(args.device) # bs * 100 * 384
        score_token_ids = None
        if args.add_score or args.relieve_CLS:
            score_token_ids = torch.tensor(test_batch['score_token_ids']).to(args.device) # 100
        for i in range(args.test_batch_size):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if score_token_ids is not None:
                        batch_score = model(
                            input_ids=input_id_list[i,:,:].to(args.device), 
                            attention_mask=attention_mask_list[i,:,:].to(args.device),
                            score_token_ids=score_token_ids[i, :].to(args.device),
                        )
                    else:
                        batch_score = model(
                            input_ids=input_id_list[i,:,:], 
                            attention_mask=attention_mask_list[i,:,:],
                        )
                    #print(batch_score.shape)
                    #print(batch_score)
                    #batch_score = batch_score[:,1].detach().cpu().tolist()
                    #batch_score = batch_score[:,0].detach().cpu().tolist()
                    #print(batch_loss)
                    batch_score_softmax = torch.softmax(batch_score[:,1176].view(-1), dim=0).detach().cpu().tolist()

                    for (q_id, d_id, b_s) in zip(query_ids[i], doc_ids[i], batch_score_softmax):
                        if q_id not in rst_dict:
                            rst_dict[q_id] = {}
                        if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                            rst_dict[q_id][d_id] = [b_s]
    return rst_dict

def main():
    ckpt_dir ='/home/jindavid/checkpoints/t5/base_new_2'
    name = 'base_new_2'

    parser = argparse.ArgumentParser()
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-test', action=om.utils.DictOrStr, default='../data_new/test_notallwithpos.json')
    parser.add_argument('-config', type=str, default='t5-base')
    parser.add_argument('-pretrained', type=str, default='t5-base')
    parser.add_argument('-tokenizer', type=str, default='t5-base')
    parser.add_argument('-checkpoint', type=str, default=ckpt_dir + '/' + name + '.bin_step-50000.bin')
    parser.add_argument('-res', type=str, default=ckpt_dir + '/' + name + '_test.trec')
    parser.add_argument('-test_batch_size', type=int, default=1)
    parser.add_argument("-doc_size", type=int, default=100)
    parser.add_argument('-metric', type=str, default='mrr_cut_10')
    parser.add_argument('-qrels', type=str, default='../data_new/test_qrel_notallwithpos.trec')
    parser.add_argument('-log_dir', type=str, default='mrr_cut_10')
    parser.add_argument("-use_global", action='store_true', default = False)
    parser.add_argument('-add_score', action='store_true', default = False)
    parser.add_argument('-add_bin', action='store_true', default = False)
    parser.add_argument('-add_rank', action='store_true', default = False)
    parser.add_argument('-max_seq_len', type=int, default=512)
    parser.add_argument('-max_query_len', type=int, default=64)
    parser.add_argument('-relieve_CLS', action='store_true', default = False)
    parser.add_argument('-number_bin', action='store_true', default = False)
    parser.add_argument('-num_global_layers', type=int, default=3)

    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=-1) # for distributed mode
    parser.add_argument( "--server_ip",type=str,default="", help="For distant debugging.",)  
    parser.add_argument( "--server_port",type=str, default="",help="For distant debugging.",)
    
    args = parser.parse_args()
    init_logger(args)
    
    

    filename = args.log_dir + 'run.log'
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)
    set_dist_args(args)
    
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer, model_max_length=512)
    tokenizer.add_tokens("[CLS]", special_tokens=True) #extra_id_-1
    bin_tokens = None
    if args.add_bin == True:
        bin_tokens = []
        for i in range(100):
            bin_tokens.append("<extra_id_{}>".format(i)) #t5 unuse token
        tokenizer.add_tokens("<extra_id_100>", special_tokens=True) #extra_id_-2
        bin_tokens.append("<extra_id_100>")

    add_rank = False
    if args.add_rank:
        add_rank = True
    if args.add_score:
        logger.info('reading test data...')
        test_set = om.t5Dataset_score(
            dataset=args.test,
            tokenizer=tokenizer,
            max_input=args.max_input,
            doc_size=args.doc_size,
            bin_tokens=bin_tokens,
            add_rank=add_rank,
            relieve_CLS=args.relieve_CLS,
            number_bin=args.number_bin,
            max_query_len=args.max_query_len,
            max_seq_len=args.max_seq_len,
        )
    else:
        logger.info('reading test data...')
        test_set = om.t5Dataset(
            dataset=args.test,
            tokenizer=tokenizer,
            max_input=args.max_input,
            doc_size=args.doc_size,
            relieve_CLS=args.relieve_CLS,
            max_query_len=args.max_query_len,
            max_seq_len=args.max_seq_len,
        )

    logger.info('loading test data...')
    
    if args.local_rank != -1:
        test_sampler = DistributedEvalSampler(test_set)
        test_loader = om.DataLoader(
            dataset=test_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=0,
            sampler=test_sampler
        )
        dist.barrier()
    else:
        test_loader = om.DataLoader(
            dataset=test_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=16,
        )
    logger.info('loading t5 model...')
    model = om.t5(
        config=args.config,
        pretrained=args.pretrained,
        doc_size=args.doc_size,
        use_global=args.use_global,
        grad_detach=False,
        num_global_layers=args.num_global_layers,
        #new_tokenizer=tokenizer, # resize for bin token
    )
    #dist.barrier()
    device = args.device
    logger.info('t5 model loading finished!')
    if args.local_rank != -1:
        state_dict = torch.load(args.checkpoint, map_location=device)
    else:
        state_dict = torch.load(args.checkpoint, map_location=device)
    state_dict = clean_dict_name(state_dict)
    model.load_state_dict(state_dict)

    logger.info('t5 state dict loading finished!')
    
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    #model.half()
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        dist.barrier()

    logger.info(args) 
    model.zero_grad()
    model.eval()   
        
    rst_dict = test(args, model, test_loader)
    dist.barrier()
    if args.local_rank != -1:
        save_trec(args.res + "_rank_{:03}".format(args.local_rank), rst_dict)
        dist.barrier()
        if args.local_rank in [-1,0]:
            merge_resfile(args.res + "_rank_*", args.res)
        dist.barrier()
    if args.local_rank in [-1,0]:
        mes = get_mrr(args.qrels, args.res, args.metric)
        logger.info("mrr@10: {}".format(mes))
        mes = get_ndcg(args.qrels, args.res, args.metric)
        logger.info("ndcg@10: {}".format(mes))
    dist.barrier()
    
    if args.local_rank != -1:
        dist.barrier()
    sys.exit(0)
if __name__ == "__main__":
    main()
