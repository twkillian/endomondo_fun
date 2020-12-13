import torch
from torch import nn, optim
from models import CNNet, TCNet, ClassTCNet
from torch.utils.data import DataLoader, Subset
from data_utils import TSData, map_data, get_npy_data, get_metadata

def load_data(args):
    raw_data, x_data, y_data = get_npy_data(args)
    train_idx, val_idx, test_idx, context = get_metadata(args)
    train_idx, val_idx, test_idx = map_data(train_idx, val_idx, test_idx, context, raw_data)
    
    if args.task == 'forecasting':
        dataset = TSData(x_data, y_data)
    elif args.task == 'prediction':
        dataset = ClassData(x_data, y_data)
    
    trainset = Subset(dataset, train_idx)
    valset = Subset(dataset, val_idx)
    testset = Subset(dataset, test_idx)
    
    trainloader = DataLoader(trainset, batch_size = args.batch_size)
    valloader = DataLoader(valset, batch_size = args.eval_batch_size)
    testloader = DataLoader(testset, batch_size = args.eval_batch_size)
    return trainloader, valloader, testloader

def get_criterion(args):
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'mse':
        criterion = nn.MSELoss()
    return criterion

def get_optimizer(args, model):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = args.weight_decay)
    return optimizer

def get_model(args):
    if args.model == 'tcn' and args.task == 'forecasting':
        num_channels = [args.n_hidden]*(args.n_blocks-1)+[len(args.y_vals)]
        num_inputs = len(args.x_vals)
        model = TCNet(num_inputs, num_channels)
    elif args.model == 'tcn' and args.task == 'prediction':
        num_channels = [args.n_hidden]*(args.n_blocks)
        num_inputs = len(args.x_vals)
        num_outputs = args.n_output_vals
        model = ClassTCNet(num_inputs, num_channels, num_outputs)
    return model

def get_scheduler(args, optimizer):
    if args.scheduler is None:
        lr_scheduler = None
    elif args.scheduler == 'plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience)
    elif args.scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)
    return lr_scheduler
