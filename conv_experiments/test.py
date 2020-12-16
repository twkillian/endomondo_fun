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
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses.append(loss.item())
        
    mean_loss = np.mean(losses)
    return mean_loss

def main(args):
    info = torch.load(args.model_path)
    args = Namespace(**info['args'])
    
    model = get_model(args)
    model.load_state_dict(info['model_state_dict'])
    model = model.to(device)
    
    criterion = get_criterion(args)
    _, valloader, testloader = load_data(args)
    mean_val_loss = evaluate(args, model, criterion, valloader)
    mean_test_loss = evaluate(args, model, criterion, testloader)
    
    print('TRAIN LOSS: ', info['mean_train_loss'])
    print('VALIDATION LOSS: ', mean_val_loss)
    print('TEST LOSS: ', mean_test_loss)
    
if __name__ == "__main__":
    args = setup_test_parser().parse_args()
    print_args(args)
    main(args)
