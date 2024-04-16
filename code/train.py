

import os
import torch
import numpy as np
from tqdm import tqdm
from parser_util import get_parser
from data_loader import MyDataset, KShotTaskSampler
from encoder import ModelManager
from transformers import AdamW, get_linear_schedule_with_warmup

from collections import defaultdict
from losses_new import MixedLoss
import json



def init_dataloader(args, mode):
    filePath = os.path.join(args.dataFile, mode + '.json')
    if mode == 'train' or mode == 'valid':
        episode_per_epoch = args.episodeTrain
    else:
        episode_per_epoch = args.episodeTest
    dataset = MyDataset(filePath)
    sampler = KShotTaskSampler(dataset, episodes_per_epoch=episode_per_epoch, n=args.numKShot, k=args.numNWay, q=args.numQShot, num_tasks=1)

    return sampler

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def init_model(args):
    device = torch.device('cuda', args.numDevice)
    torch.cuda.set_device(device)
    model = ModelManager(args).to(device)
    return model

def init_optim(args, model):

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    


    return optimizer

def init_lr_scheduler(args, optim):
    '''
    Initialize the learning rate scheduler
    '''
    t_total = args.epochs * args.episodeTrain
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    return scheduler

def deal_data(support_set, query_set, episode_labels):

    support_text, query_text, support_aug, query_aug = [], [], [], []
    
    for x in support_set:
        support_text.append(x["text"])
        support_aug.append(x["aug_text"])
    for x in query_set:
        query_text.append(x["text"])
        query_aug.append(x["aug_text"])


    return support_text, query_text, support_aug, query_aug


def train(args, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):

    if val_dataloader is None:
        # best_state = None
        acc_best_state = None
        f1_best_state = None 
    train_loss, epoch_train_loss = [], []
    train_acc, epoch_train_acc = [], []
    val_loss, epoch_val_loss = [], []
    val_acc, epoch_val_acc = [], []
   
    best_acc = 0
    loss_fn = MixedLoss(beta=args.beta, gamma=args.gamma)
    
    
    acc_best_model_path = os.path.join(args.fileModelSave, 'acc_best_model.pth')
    last_model_path = os.path.join(args.fileModelSave, 'last_model.pth')

    tolerate = 0
    for epoch in range(args.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        model.train()
        if tolerate >= args.tolerate:
            break
        for batch in tqdm(tr_dataloader):
            optim.zero_grad()
            
            support_set, query_set, episode_labels = batch
            support_text, query_text, support_aug, query_aug = deal_data(support_set, query_set, episode_labels)
           
           
            tasks = defaultdict(list)
            tasks[0] = support_text + query_text

            repeat = 8
            for i, aug_text in enumerate(support_aug):
                glen = len(aug_text)
                for j in range(glen):
                    tasks[j+1].append(aug_text[j])
                if glen < repeat:
                    for j in range(glen, repeat):
                        tasks[j+1].append(support_text[i])
            
            for i, aug_text in enumerate(query_aug):
                glen = len(aug_text)
                for j in range(glen):
                    tasks[j+1].append(aug_text[j])
                if glen < repeat:
                    for j in range(glen, repeat):
                        tasks[j+1].append(query_text[i])
            
            
            tasks_em = []
            for i in range(repeat+1):
                model_outputs = model(tasks[i], episode_labels)
                tasks_em.append(model_outputs)
            tasks_em = torch.stack(tasks_em)

          
            loss, acc= loss_fn(tasks_em, 
                                nway=args.numNWay, 
                                kshot=args.numKShot,
                                qshot=args.numQShot,
                                repeat=repeat,model=model)
           

            loss.backward()
            
            optim.step()
            lr_scheduler.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            
        avg_loss = np.mean(train_loss[-args.episodeTrain:])
        avg_acc = np.mean(train_acc[-args.episodeTrain:])
      
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        epoch_train_loss.append(avg_loss)
        epoch_train_acc.append(avg_acc)
        

       

        if val_dataloader is None:
            continue
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader):

                support_set, query_set, episode_labels = batch
                support_text, query_text, support_aug, query_aug = deal_data(support_set, query_set, episode_labels)
           
              
                tasks = defaultdict(list)
                tasks[0] = support_text + query_text

                repeat = 8
                for i, aug_text in enumerate(support_aug):
                    glen = len(aug_text)
                    for j in range(glen):
                        tasks[j+1].append(aug_text[j])
                    if glen < repeat:
                        for j in range(glen, repeat):
                            tasks[j+1].append(support_text[i])
            
                for i, aug_text in enumerate(query_aug):
                    glen = len(aug_text)
                    for j in range(glen):
                        tasks[j+1].append(aug_text[j])
                    if glen < repeat:
                        for j in range(glen, repeat):
                            tasks[j+1].append(query_text[i])
            
            
                tasks_em = []
                for i in range(repeat+1):
                    model_outputs = model(tasks[i], episode_labels)
                    tasks_em.append(model_outputs)
                tasks_em = torch.stack(tasks_em)
               
                loss, acc = loss_fn(
                                tasks_em, 
                                nway=args.numNWay, 
                                kshot=args.numKShot,
                                qshot=args.numQShot,
                                repeat=repeat,model=model)
                
                val_loss.append(loss.item())
                val_acc.append(acc.item())
               
        avg_loss = np.mean(val_loss[-args.episodeTrain:])
        avg_acc = np.mean(val_acc[-args.episodeTrain:])
        

        epoch_val_loss.append(avg_loss)
        epoch_val_acc.append(avg_acc)
       

        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
       
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), acc_best_model_path)
            best_acc = avg_acc
            acc_best_state = model.state_dict()
            tolerate = 0
        else:
            tolerate += 1

       

    torch.save(model.state_dict(), last_model_path)

    for name in ['epoch_train_loss', 'epoch_train_acc', 'epoch_val_loss', 'epoch_val_acc']:
        save_list_to_file(os.path.join(args.fileModelSave,
                                       name + '.txt'), locals()[name])

    return acc_best_state
    

def test(args, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    
    val_acc = []
    val_loss = []
    val_f1 = []
    loss_fn = MixedLoss( beta=args.beta, gamma=args.gamma)
    original_acc = []
    entailment_acc = []
    neutral_acc = []
    contradiction_acc = []
    model.eval()
    with torch.no_grad():
        

        for batch in tqdm(test_dataloader):

            support_set, query_set, episode_labels = batch
            support_text, query_text, support_aug, query_aug = deal_data(support_set, query_set, episode_labels)
        
         
            tasks = defaultdict(list)
            tasks[0] = support_text + query_text

            repeat = 8
            for i, aug_text in enumerate(support_aug):
                glen = len(aug_text)
                for j in range(glen):
                    tasks[j+1].append(aug_text[j])
                if glen < repeat:
                    for j in range(glen, repeat):
                        tasks[j+1].append(support_text[i])
            
            for i, aug_text in enumerate(query_aug):
                glen = len(aug_text)
                for j in range(glen):
                    tasks[j+1].append(aug_text[j])
                if glen < repeat:
                    for j in range(glen, repeat):
                        tasks[j+1].append(query_text[i])
        
            
            tasks_em = []
            for i in range(repeat+1):
                model_outputs = model(tasks[i], episode_labels)
                tasks_em.append(model_outputs)
            tasks_em = torch.stack(tasks_em)
          
            loss, acc = loss_fn(
                                tasks_em, 
                                nway=args.numNWay, 
                                kshot=args.numKShot,
                                qshot=args.numQShot,
                                repeat=repeat,model=model)
        
            

            val_loss.append(loss.item())
            val_acc.append(acc.item())
           
            

    avg_acc = np.mean(val_acc)
    avg_loss = np.mean(val_loss)
   

    print('Test Acc: {}'.format(avg_acc))
    print('Test Loss: {}'.format(avg_loss))
   

 
    path = args.fileModelSave + "/test_score.json"
    with open(path, "a+") as fout:
        tmp = {"Acc": avg_acc, "Loss": avg_loss}

        fout.write("%s\n" % json.dumps(tmp, ensure_ascii=False))

    return avg_acc, avg_loss


def write_args_to_josn(args):
    path = args.fileModelSave + "/config.json"
    args = vars(args)
    json_str = json.dumps(args, indent=4)
    with open(path, 'w') as json_file:
        json_file.write(json_str)
        

def main():
    args = get_parser().parse_args()
    
    if not os.path.exists(args.fileModelSave):
        os.makedirs(args.fileModelSave)

    write_args_to_josn(args)

    model = init_model(args)
    print(model)

    tr_dataloader = init_dataloader(args, 'train')
    val_dataloader = init_dataloader(args, 'valid')
    test_dataloader = init_dataloader(args, 'test')
   

    optim = init_optim(args, model)
    lr_scheduler = init_lr_scheduler(args, optim)
    results = train(args=args,
                    tr_dataloader=tr_dataloader,
                    val_dataloader=val_dataloader,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler)
    acc_best_state = results

    model.load_state_dict(torch.load(args.fileModelSave + "/last_model.pth"))
    print('Testing with last model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(torch.load(args.fileModelSave + "/acc_best_model.pth"))
    print('Testing with acc best model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model)

   
   
if __name__ == '__main__':
    main()
