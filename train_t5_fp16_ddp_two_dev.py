import sys
import argparse
import torch
import torch.nn as nn

import src as om
from src.utils import init_logger, optimizer_to, save_trec, get_mrr, set_dist_args, merge_resfile, DistributedEvalSampler, ListwiseLoss, clean_dict_name

from transformers import get_linear_schedule_with_warmup, T5Tokenizer
torch.multiprocessing.set_sharing_strategy('file_system')

import logging

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from contextlib import nullcontext

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def dev(args, model, dev_loader):

    rst_dict = {}
    for dev_batch in dev_loader:
        query_ids, doc_ids, labels = dev_batch['query_ids'], dev_batch['doc_ids'], dev_batch['labels']

        input_id_list = dev_batch['input_ids'] # bs * 100 * 384
        attention_mask_list = dev_batch['attention_mask'] # bs * 100 * 384
        score_token_ids = None
        score_memory = None
        if args.add_score and args.score_embedding:
            score_memory = torch.tensor(dev_batch['score_memory']).to(args.device) # 100
        if args.add_score or args.relieve_CLS:
            score_token_ids = torch.tensor(dev_batch['score_token_ids']).to(args.device) # 100
        for i in range(args.dev_batch_size):

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    batch_score = model(
                        input_ids=input_id_list[i,:,:], 
                        attention_mask=attention_mask_list[i,:,:],
                        score_token_ids=score_token_ids[i,:] if score_token_ids is not None else None,
                        score_memory=score_memory[i,:] if score_memory is not None else None,
                    )
                    
                    if args.loss == 'BCE':
                        batch_score = batch_score[:,0].detach().cpu().tolist()
                    elif args.loss == 'CE' or args.loss == 'list-wise':
                        #batch_score = batch_score[:,1].detach().cpu().tolist()
                        batch_score = batch_score[:,1176].detach().cpu().tolist()
                        #batch_score_softmax = torch.softmax(batch_score[:,1].view(-1), dim=0).detach().cpu().tolist()
                        #batch_score_softmax = torch.softmax(batch_score[:,1176].view(-1), dim=0).detach().cpu().tolist()

                    for (q_id, d_id, b_s, l) in zip(query_ids[i], doc_ids[i], batch_score, labels[i]):
                        if q_id not in rst_dict:
                            rst_dict[q_id] = {}
                        if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                            rst_dict[q_id][d_id] = [b_s, l]

    return rst_dict, batch_score
    
def train(args, logger, model, m_optim, m_scheduler, train_loader, dev_loader, test_loader, loss_fn, train_sampler=None):

    writer = SummaryWriter(log_dir=args.log_dir)
    best_mes = 0.0
    best_mes_test = 0.0
    global_step = 0
    avg_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epoch):
        logger.info('start epoch for {}'.format(epoch))
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)
        
        for step, train_batch in enumerate(train_loader):

            input_id_list = train_batch['input_ids'].to(args.device) # 100 * 384
            attention_mask_list = train_batch['attention_mask'].to(args.device) # 100 * 384
            label_list = torch.tensor(train_batch['raw_labels']).to(args.device) # 100
            score_token_ids = None
            score_memory = None
            if args.add_score and args.score_embedding:
                score_memory = torch.tensor(train_batch['score_memory']).to(args.device) # 100
            if args.add_score or args.relieve_CLS:
                score_token_ids = train_batch['score_token_ids'].to(args.device) # 100
            for i in range(args.batch_size):

                with torch.cuda.amp.autocast():
                    sync_context = model.no_sync if (args.local_rank != -1 and (step+1) % args.gradient_accumulation_steps != 0) else nullcontext
                    with sync_context():
                        batch_score = model(
                            input_ids=input_id_list[i,:,:], 
                            attention_mask=attention_mask_list[i,:,:],
                            score_token_ids=score_token_ids[i,:] if score_token_ids is not None else None,
                            score_memory=score_memory[i,:] if score_memory is not None else None,
                        )
                        if args.loss.lower() == "bce":

                            label_tensor = label_list[i,:args.doc_size].repeat(args.doc_size, 1)
                            label_tensor = label_tensor.to(args.device)
                            mask = label_tensor - label_tensor.t()

                            score_tensor = batch_score[:args.doc_size,0].squeeze(-1).repeat(args.doc_size, 1)
                            diff_tensor = score_tensor - score_tensor.t()
                            diff_score = diff_tensor[mask > 0]

                            batch_loss = loss_fn(torch.sigmoid(diff_score), torch.ones(diff_score.size()).to(args.device))

                        elif args.loss.lower() == "ce":
                            #print(batch_score[:args.doc_size,:])
                            #print(label_list[i,:args.doc_size])
                            #print(batch_score[:args.doc_size,[6136, 1176]])
                            #batch_loss = loss_fn(batch_score[:args.doc_size,:], label_list[i,:args.doc_size])
                            batch_loss = loss_fn(batch_score[:args.doc_size,[6136, 1176]], label_list[i,:args.doc_size])
                        elif args.loss.lower() == "list-wise":
                            batch_loss = loss_fn(batch_score[:args.doc_size,[6136, 1176]], label_list[i,:args.doc_size])
                        if args.n_gpu > 1:
                            batch_loss = batch_loss.mean()
                        if args.gradient_accumulation_steps > 1:
                            batch_loss = batch_loss / args.gradient_accumulation_steps

                avg_loss += batch_loss.item()
                with sync_context():
                    scaler.scale(batch_loss).backward()    

            # logging train and evaluation
            if (step+1) % args.gradient_accumulation_steps == 0:
                
                scaler.unscale_(m_optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scaler.step(m_optim)
                scaler.update()
                m_optim.zero_grad()
                m_scheduler.step()

                if args.logging_step > 0 and ((global_step+1) % args.logging_step == 0):
                    dist.barrier()
                    avg_loss /= args.logging_step*args.batch_size
                    if args.local_rank in [-1, 0]:
                        logger.info("global step: {}, local step: {}, loss: {}".format(global_step+1, (step+1) * args.world_size, avg_loss))
                        writer.add_scalar("loss", avg_loss, global_step)                        
                    dist.barrier()
                    avg_loss = 0.0

                if (global_step+1) % args.eval_every == 0:
                    
                    model.eval()
                    dist.barrier()
                    rst_dict, _ = dev(args, model, dev_loader)

                    rst_dict_test, _ = dev(args, model, test_loader)
                    dist.barrier()

                    model.train()

                    if args.local_rank != -1:
                        save_trec(args.res + "_rank_{:03}".format(args.local_rank), rst_dict)
                        dist.barrier()
                        save_trec(args.res_test + "_rank_{:03}".format(args.local_rank), rst_dict_test)
                        if args.local_rank in [-1,0]:
                            merge_resfile(args.res + "_rank_*", args.res + "_step-{}".format(global_step+1))
                        dist.barrier()
                        if args.local_rank in [-1,0]:
                            merge_resfile(args.res_test + "_rank_*", args.res_test + "_step-{}".format(global_step+1))
                        dist.barrier()
                    else:
                        save_trec(args.res + "_step-{}".format(global_step+1), rst_dict)
                        save_trec(args.res_test + "_step-{}".format(global_step+1), rst_dict_test)
                    mes = get_mrr(args.qrels, args.res + "_step-{}".format(global_step+1), args.metric)
                    mes_test = get_mrr(args.qrels_test, args.res_test + "_step-{}".format(global_step+1), args.metric)
                    saved = False
                    if mes >= best_mes:
                        best_mes = mes
                    if mes_test >= best_mes_test:
                        best_mes_test = mes_test 
                    """if mes >= best_mes:
                        best_mes = mes
                        logger.info('Saving best model at step {}'.format(global_step+1))
                        torch.save(model.state_dict(), args.save + "_step-{}.bin".format(global_step+1))
                        saved=True"""

                    #if (global_step+1) % (4 * args.eval_every / args.gradient_accumulation_steps) == 0:
                    if (global_step+1) % args.eval_every == 0:
                        logger.info('Saving model at step {}'.format(global_step+1))
                        torch.save(model.state_dict(), args.save + "_step-{}.bin".format(global_step+1))

                    saved = False
                    logger.info("global step: {}, messure: {}, best messure: {}, test messure: {}, test best messure: {}".format(global_step+1, mes, best_mes, mes_test, best_mes_test))

                    writer.add_scalar('dev', mes, global_step)
                    writer.add_scalar('test', mes_test, global_step)
                
                global_step += 1

    return 
        

def main():
    ckpt="t5-base"
    parser = argparse.ArgumentParser()

    # training setup
    parser.add_argument('-optimizer', type=str, default='adamw')
    parser.add_argument("-doc_size", type=int, default = 10)
    parser.add_argument("-use_global", action='store_true', default = False)
    parser.add_argument("-grad_detach", action='store_true', default = False)
    parser.add_argument('-config', type=str, default=ckpt)
    parser.add_argument('-tokenizer', type=str, default=ckpt)
    parser.add_argument('-pretrained', type=str, default=ckpt)
    parser.add_argument('-loss', type=str, default="ce")

    # ddp
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=-1) # for distributed mode
    parser.add_argument( "--server_ip",type=str,default="", help="For distant debugging.",)  
    parser.add_argument( "--server_port",type=str, default="",help="For distant debugging.",)
    # dataset
    parser.add_argument('-train', action=om.utils.DictOrStr, default='../data_new/train_366000.json')
    parser.add_argument('-dev', action=om.utils.DictOrStr, default='../data_new/dev_914.json')
    parser.add_argument('-test', action=om.utils.DictOrStr, default='../data_new/dev_914.json')
    parser.add_argument('-qrels', type=str, default='/home/jindavid/data/msmarco-docdev-qrels.tsv')
    parser.add_argument('-qrels_test', type=str, default='/home/jindavid/data/msmarco-docdev-qrels.tsv')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-max_query_len', type=int, default=64)
    parser.add_argument('-max_seq_len', type=int, default=512)

    # training parameters
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-lr', type=float, default=5e-4)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument("-max_steps", type=int)
    parser.add_argument('-gradient_accumulation_steps', type=int, default=1) 
    parser.add_argument("-max_grad_norm", default=1.0,type=float, help="Max gradient norm.")

    # logging and saving
    parser.add_argument('-eval_every', type=int, default=10000)
    parser.add_argument('-dev_batch_size', type=int, default=1)
    parser.add_argument('-logging_step', type=int, default=100)
    parser.add_argument("-log_dir", type=str)
    parser.add_argument('-save', type=str, default='./checkpoints/t5.bin')
    parser.add_argument('-res', type=str, default='ru/home/jindavid/checkpoints/$Namens/t5.trec')
    parser.add_argument('-res_test', type=str, default='ru/home/jindavid/checkpoints/$Namens/t5.trec')
    parser.add_argument('-metric', type=str, default='mrr_cut_100')
    parser.add_argument('-num_global_layers', type=int, default=3)
    parser.add_argument('-retraining', action='store_true', default=False)
    parser.add_argument('-add_score', action='store_true', default = False)
    parser.add_argument('-add_rank', action='store_true', default = False)
    parser.add_argument('-add_bin', action='store_true', default = False)
    parser.add_argument('-relieve_CLS', action='store_true', default = False)
    parser.add_argument('-number_bin', action='store_true', default = False)
    parser.add_argument('-score_embedding', action='store_true', default = False)
    
    args = parser.parse_args()
    init_logger(args)
    device = args.device

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
    bin_tokens = []
    for i in range(101):
        bin_tokens.append("{}".format(i))
    logger.info('reading training data...')
    if args.add_score:
        train_set = om.t5Dataset_score(
            dataset=args.train,
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
            
        logger.info('reading dev data...')
        dev_set = om.t5Dataset_score(
            dataset=args.dev,
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
        train_set = om.t5Dataset(
            dataset=args.train,
            tokenizer=tokenizer,
            max_input=args.max_input,
            doc_size=args.doc_size,
            relieve_CLS=args.relieve_CLS,
            max_query_len=args.max_query_len,
            max_seq_len=args.max_seq_len,
        )
            
        logger.info('reading dev data...')
        dev_set = om.t5Dataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            doc_size=args.doc_size,
            max_input=args.max_input,
            relieve_CLS=args.relieve_CLS,
            max_query_len=args.max_query_len,
            max_seq_len=args.max_seq_len,
        )

        logger.info('reading test data...')
        test_set = om.t5Dataset(
            dataset=args.test,
            tokenizer=tokenizer,
            doc_size=args.doc_size,
            max_input=args.max_input,
            relieve_CLS=args.relieve_CLS,
            max_query_len=args.max_query_len,
            max_seq_len=args.max_seq_len,
        )

    logger.info('loading train data...')
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_set)
        train_loader = om.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            sampler=train_sampler
        )
        dev_sampler = DistributedEvalSampler(dev_set)
        logger.info('loading dev data...')
        dev_loader = om.DataLoader(
            dataset=dev_set,
            batch_size=args.dev_batch_size,
            shuffle=False,
            num_workers=8,
            sampler=dev_sampler
        )

        test_sampler = DistributedEvalSampler(test_set)
        logger.info('loading test data...')
        test_loader = om.DataLoader(
            dataset=test_set,
            batch_size=args.dev_batch_size,
            shuffle=False,
            num_workers=8,
            sampler=test_sampler
        )
        dist.barrier()
    else:
        train_loader = om.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
        )
        
        logger.info('loading dev data...')
        dev_loader = om.DataLoader(
            dataset=dev_set,
            batch_size=args.dev_batch_size,
            shuffle=True,
            num_workers=8,
        )

        logger.info('loading test data...')
        test_loader = om.DataLoader(
            dataset=test_set,
            batch_size=args.dev_batch_size,
            shuffle=True,
            num_workers=8,
        )
    logger.info('loading t5 model...')
    model = om.t5(
        pretrained=args.pretrained,
        config=args.config,
        doc_size=args.doc_size,
        use_global=args.use_global,
        num_global_layers=args.num_global_layers,
        grad_detach=args.grad_detach,
        #new_tokenizer=tokenizer, # resize for bin token
    )
    
    dist.barrier()
    if args.retraining:
        if args.local_rank != -1:
            state_dict = torch.load(args.pretrained, map_location='cuda:{}'.format(args.local_rank))
        else:
            state_dict = torch.load(args.pretrained, map_location='cuda:0')
        state_dict = clean_dict_name(state_dict)
        model.load_state_dict(state_dict)
    
    model.init_position(args.score_embedding)
    logger.info('Loading finished!')

    if args.loss.lower() == "bce":
        
        loss_fn = nn.BCELoss()

    elif args.loss.lower() == "ce":

        loss_fn = nn.CrossEntropyLoss()
    elif args.loss.lower() == "list-wise":

        loss_fn = ListwiseLoss()
    loss_fn.to(device)
    
    model.to(device)
    #model.half()

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        loss_fn = nn.DataParallel(loss_fn)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        dist.barrier()
    model.zero_grad()
    model.train()

    #for key, param in model.named_parameters():
    #    #print(key)
    #    if 'position_memory' not in key and 'relative_attention_bias' not in key:
    #        param.requires_grad = False

    if args.optimizer.lower() == 'adam':
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        m_optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    optimizer_to(m_optim, device)
    m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//(args.batch_size*args.gradient_accumulation_steps) if args.max_steps is None else args.max_steps)

    ### start training ###
    logger.info(args)
    train(args, logger, model, m_optim, m_scheduler, train_loader, dev_loader, test_loader, loss_fn, train_sampler=train_sampler)
    if args.local_rank != -1:
        dist.barrier()
if __name__ == "__main__":
    main()