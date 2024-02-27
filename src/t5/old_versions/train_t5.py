import sys
import argparse
import torch
import torch.nn as nn

import src as om
from src.utils import init_logger, optimizer_to, save_trec, get_mrr

from transformers import get_linear_schedule_with_warmup, T5Tokenizer
torch.multiprocessing.set_sharing_strategy('file_system')

import logging

from torch.utils.tensorboard import SummaryWriter

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def dev(args, model, dev_loader):

    rst_dict = {}
    for dev_batch in dev_loader:
        query_ids, doc_ids, labels = dev_batch['query_ids'], dev_batch['doc_ids'], dev_batch['labels']

        input_id_list = dev_batch['input_ids'] # bs * 100 * 384
        attention_mask_list = dev_batch['attention_mask'] # bs * 100 * 384

        for i in range(args.dev_batch_size):

            with torch.no_grad():
                batch_score = model(
                    input_ids=input_id_list[i,:,:].to(args.device), 
                    attention_mask=attention_mask_list[i,:,:].to(args.device),
                )
                
                batch_score = batch_score[:,0].detach().cpu().tolist()

                for (q_id, d_id, b_s, l) in zip(query_ids[i], doc_ids[i], batch_score, labels[i]):
                    if q_id not in rst_dict:
                        rst_dict[q_id] = {}
                    if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                        rst_dict[q_id][d_id] = [b_s, l]

    return rst_dict, batch_score
    
    
def train(args, logger, model, m_optim, m_scheduler, train_loader, dev_loader, loss_fn):

    writer = SummaryWriter(log_dir=args.log_dir)
    best_mes = 0.0
    global_step = 0

    for epoch in range(args.epoch):

        avg_loss = 0.0
        for step, train_batch in enumerate(train_loader):

            input_id_list = train_batch['input_ids'].to(args.device) # 100 * 384
            attention_mask_list = train_batch['attention_mask'].to(args.device) # 100 * 384
            label_list = torch.tensor(train_batch['labels']).to(args.device) # 100

            for i in range(args.batch_size):

                batch_score = model(
                    input_ids=input_id_list[i,:,:], 
                    attention_mask=attention_mask_list[i,:,:],
                )
                # BCE loss
                if args.loss.lower() == "bce":

                    label_tensor = label_list[i,:args.doc_size].repeat(args.doc_size, 1)
                    label_tensor = label_tensor.to(args.device)
                    mask = label_tensor - label_tensor.t()

                    score_tensor = batch_score[:args.doc_size,0].squeeze(-1).repeat(args.doc_size, 1)
                    diff_tensor = score_tensor - score_tensor.t()
                    diff_score = diff_tensor[mask > 0]

                    batch_loss = loss_fn(torch.sigmoid(diff_score), torch.ones(diff_score.size()).to(args.device))

                elif args.loss.lower() == "ce":

                    batch_loss = loss_fn(batch_score[:args.doc_size,:], label_list[i,:args.doc_size])
                
                if args.gradient_accumulation_steps > 1:
                    batch_loss = batch_loss / args.gradient_accumulation_steps
                avg_loss += batch_loss.item()

                batch_loss.backward()

            # logging train and evaluation
            if (step+1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                m_optim.step()
                m_scheduler.step()
                m_optim.zero_grad()

                if args.logging_step > 0 and ((global_step+1) % args.logging_step == 0):
                    avg_loss /= args.logging_step*args.batch_size
                    logger.info("global step: {}, local step: {}, loss: {}".format(global_step+1, step+1, avg_loss))
                    writer.add_scalar("loss", avg_loss, global_step)                        
                    avg_loss = 0.0

                if (global_step+1) % args.eval_every == 0:
                    
                    model.eval()
                    rst_dict, _ = dev(args, model, dev_loader)
                    model.train()

                    save_trec(args.res, rst_dict)
                    mes = get_mrr(args.qrels, args.res, args.metric)
                    best_mes = mes if mes >= best_mes else best_mes

                    logger.info('Saving model at step {}'.format(global_step+1))
                    torch.save(model.state_dict(), args.save + "_step-{}.bin".format(global_step+1))
                    logger.info("global step: {}, messure: {}, best messure: {}".format(global_step+1, mes, best_mes))

                    writer.add_scalar('dev', mes, step)
                
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
    parser.add_argument('-vocab', type=str, default=ckpt)
    parser.add_argument('-pretrained', type=str, default=ckpt)
    parser.add_argument('-loss', type=str, default="ce")
    
    # dataset
    parser.add_argument('-train', action=om.utils.DictOrStr, default='../data_new/train_366000.json')
    parser.add_argument('-dev', action=om.utils.DictOrStr, default='../data_new/dev_914.json')
    parser.add_argument('-qrels', type=str, default='/home/jindavid/data/msmarco-docdev-qrels.tsv')
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
    parser.add_argument('-metric', type=str, default='mrr_cut_100')

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

    tokenizer = T5Tokenizer.from_pretrained(args.vocab, model_max_length=512)

    logger.info('reading training data...')
    train_set = om.t5Dataset(
        dataset=args.train,
        tokenizer=tokenizer,
        max_input=args.max_input
    )
        
    logger.info('reading dev data...')
    dev_set = om.t5Dataset(
        dataset=args.dev,
        tokenizer=tokenizer,
        max_input=args.max_input,
    )

    logger.info('loading train data...')
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
    
    logger.info('loading t5 model...')
    model = om.t5(
        pretrained=args.pretrained,
        doc_size=args.doc_size,
        use_global=args.use_global,
        num_global_layers=3,
        grad_detach=args.grad_detach,
    )
    logger.info('Loading finished!')

    if args.loss.lower() == "bce":
        
        loss_fn = nn.BCELoss()

    elif args.loss.lower() == "ce":

        loss_fn = nn.CrossEntropyLoss()

    loss_fn.to(device)
    
    model.to(device)
    model.zero_grad()
    model.train()

    if args.optimizer.lower() == 'adam':
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        m_optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    optimizer_to(m_optim, device)
    m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//(args.batch_size*args.gradient_accumulation_steps) if args.max_steps is None else args.max_steps)

    ### start training ###
    logger.info(args)
    train(args, logger, model, m_optim, m_scheduler, train_loader, dev_loader, loss_fn)
    
if __name__ == "__main__":
    main()