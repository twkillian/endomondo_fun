import torch
from torch import nn
import numpy as np
from argparse import Namespace
from utils import get_model, get_criterion, load_data
from parameters import print_args, setup_test_parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(args, model, criterion, loader):
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses.append(loss.item())

            if args.loss == 'bce':
                label_preds = torch.sigmoid(y_pred).round()
                correct += (label_preds == y).sum().item()
            
            total += x.shape[0]

    mean_acc = correct / total
    mean_loss = np.mean(losses)
    return mean_loss, mean_acc

def main(args):
    info = torch.load(args.model_path)
    train_args = Namespace(**info['args'])
    
    model = get_model(train_args)
    model.load_state_dict(info['model_state_dict'])
    model = model.to(device)
    
    criterion = get_criterion(train_args)
    trainloader, valloader, testloader = load_data(train_args)
    mean_train_loss, train_acc = evaluate(train_args, model, criterion, trainloader)
    mean_val_loss, val_acc = evaluate(train_args, model, criterion, valloader)
    mean_test_loss, test_acc = evaluate(train_args, model, criterion, testloader)
    
    print('TRAIN LOSS: ', mean_train_loss)
    print('TRAIN ACC: ', train_acc)
    print('VALIDATION LOSS: ', mean_val_loss)
    print('VALIDATION ACC: ', val_acc)
    print('TEST LOSS: ', mean_test_loss)
    print('TEST ACC: ', test_acc)
    
if __name__ == "__main__":
    args = setup_test_parser().parse_args()
    print_args(args)
    main(args)
