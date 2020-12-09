import torch
from torch import nn, optim
import os
import numpy as np
from utils import get_model, get_criterion, get_optimizer, get_scheduler, load_data
from parameters import setup_parser, print_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def checkpoint_save(model, optimizer, lr_scheduler, epoch, CHECKPOINT_PATH):
    save_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
                }
    if lr_scheduler is not None:
        save_dict['scheduler_state_dict'] = lr_scheduler.state_dict()
    torch.save(save_dict, CHECKPOINT_PATH)

def checkpoint_load(model, optimizer, lr_scheduler, CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    return model, optimizer, lr_scheduler, start_epoch
    

def evaluate(args, model, criterion, valloader):
    model.eval()
    losses = []
    correct = 0.
    total = len(valloader.dataset)
    with torch.no_grad():
        for x,y in valloader:
            x = x.to(device)
            y = y.to(device)
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses.append(loss.item())
            
            if args.loss == 'ce':
                preds = y_pred.cpu().numpy().argmax(axis=1)
            else:
                preds = y_pred.cpu().numpy()
                
            targets = y.cpu().numpy().reshape(preds.shape)
            correct += (preds == targets).sum()
    
    eval_acc = correct / total
    mean_eval_loss = np.mean(losses)
    return mean_eval_loss, eval_acc

def train_epoch(args, model, optimizer, criterion, trainloader):
    losses = []
    model.train()
    for x, y in trainloader:
        x = x.to(device)
        y = y.to(device)
        
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    mean_train_loss = np.mean(losses)
    return mean_train_loss

def train_log(args, epoch, model, criterion, valloader, mean_train_loss):
    print('Epoch ', epoch, ' metrics:')
    metrics = {}
    metrics['mean_train_loss'] = mean_train_loss
    if epoch % args.eval_interval == 0:
        mean_eval_loss, eval_acc = evaluate(args, model, criterion, valloader)
        metrics['mean_eval_loss'] = mean_eval_loss
        metrics['eval_acc'] = eval_acc
    metric_output = ', '.join([f'{key} : {value}' for key, value in metrics.items()])
    print(metric_output)
    
def main(args):
    model = get_model(args).to(device)
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_scheduler(args, optimizer)
    criterion = get_criterion(args)
    start_epoch = 0
    
    trainloader, valloader, testloader = load_data(args)
    
    CHECKPOINT_PATH = f'{args.checkpoint_dir}/checkpoint.tar'
    
    if os.path.exists(CHECKPOINT_PATH):
        model, optimizer, lr_scheduler, start_epoch = checkpoint_load(model, optimizer, lr_scheduler, CHECKPOINT_PATH)
    
    print('Started training!')
    for epoch in range(start_epoch, args.n_epochs):
        if lr_scheduler is not None:
            lr_scheduler.step()
        mean_train_loss = train_epoch(args, model, optimizer, criterion, trainloader)
        train_log(args, epoch, model, criterion, valloader, mean_train_loss)
        if epoch % args.checkpoint_interval == 0:
            checkpoint_save(model, optimizer, lr_scheduler, epoch, CHECKPOINT_PATH)
    
    if args.final_save_fpath is not None:
        torch.save({
                    'args': vars(args),
                    'model_state_dict': model.state_dict()
                   }, args.final_save_fpath)
    

if __name__ == '__main__':
    args = setup_parser().parse_args()
    print_args(args)
    main(args)
