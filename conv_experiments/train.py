import torch
from torch import nn, optim
import os
import argparse
from utils import get_model, get_criterion, get_optimizer, get_scheduler, load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_parser():
    parser = argparse.ArgumentParser(prog = 'TCN&CNN', description = 'Convolution approaches for time series forecasting and prediction')
    parser = argparse.add_argument('--dataset_path', type=str, default = '/scratch/hdd001/home/dullerud/endomondo')
    parser.add_argument('--task', choices = ['prediction', 'forecasting'], default='forecasting', help = 'Task on which to train model')
    parser.add_argument('--y_val', nargs='+', type=str, default=None, help='Output values used to train model')
    parser.add_argument('--model', choices=['cnn', 'tcn'], default = 'cnn', help = 'Model architecture to use for training')
    parser.add_argument('--n_series', type=str, default = 2, help = 'Number of time series in multivariate analysis')
    parser.add_argument('--n_hidden', type=int, default = 2)
    parser.add_argument('--n_blocks', type=int, default=8, help='# of temporal blocks (default: 4)')
    parser.add_argument('--pooling_strategy', choices=['max', 'avg'], default = 'max', help = 'Pooling strategy to use in model architecture')
    parser.add_argument('--n_epochs', type=int, default = 250, help = 'Number of epochs to train')
    parser.add_argument('--optimizer', type=str, default = 'adam', help = 'Optimizer to use for training model')
    parser.add_argument('--lr', type=float, default = 0.005, help = 'Learning rate to use for training model')
    parser.add_argument('--weight_decay', type=float, default = 0., help = 'Weight decay to use for training model')
    parser.add_argument('--scheduler', type=str, default = None, help = 'Learning rate scheduler to use for optimizer')
    parser.add_argument('--milestones', nargs = '+', type=int, help = 'Milestone for MultistepLR')
    parser.add_argument('--patience', type=int, default = 10, help = 'Patience for ReduceLROnPlateau')
    parser.add_argument('--loss', type=str, default = 'mse', help = 'Loss to use for training model')
    parser.add_argument('--batch_size', type=int, default = 128, help = 'Batch size to use for training model')
    parser.add_argument('--eval_batch_size', type=int, default = 128, help = 'Batch size to use for evaluating model')
    parser.add_argument('--eval_interval', type=int, default = 50, help = 'Interval size for number of epochs to evaluate model')
    parser.add_argument('--checkpoint_interval', type=int, default = 25, help = 'Interval size for number of epochs to save checkpoint model')
    parser.add_argument('--checkpoint_dir', type=str, default = '.', help = 'Checkpoint directory')
    parser.add_argument('--final_save_fpath', type=str, default = 'saved_models/model.tar', help = 'Final filepath in which to save model')
    return parser

def checkpoint_save(model, optimizer, lr_scheduler, epoch, CHECKPOINT_PATH):
    save_dict = {
                'model_state_dict': model.state_dict()
                'optimizer_state_dict': optimizer.state_dict()
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
        for x,y in valloader():
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
    model = get_model(args)
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_scheduler(args, optimizer)
    criterion = get_criterion(args)
    start_epoch = 0
    
    trainloader, valloader, testloader = load_data(args)
    
    CHECKPOINT_PATH = f'{args.checkpoint_dir}/checkpoint.tar'
    
    if os.path.exists(CHECKPOINT_PATH):
        model, optimizer, lr_scheduler, start_epoch = checkpoint_load(model, optimizer, lr_scheduler, CHECKPOINT_PATH)
    
    for epoch in range(start_epoch, args.n_epochs):
        if lr_scheduler is not None:
            lr_scheduler.step()
        mean_train_loss = train_epoch(args, model, optimizer, criterion, trainloader, valloader)
        train_log(args, epoch, model, criterion, valloader, mean_train_loss)
        if epoch % args.checkpoint_interval == 0:
            checkpoint_save(model, optimizer, lr_scheduler, epoch, CHECKPOINT_PATH)
    
    if args.final_save_fpath is not None:
        torch.save({
                    'args': vars(args)
                    'model_state_dict': model.state_dict()
                   }, args.final_save_fpath)
    

if __name__ == '__main__':
    args = setup_parser().parse_args()
    main(args)
