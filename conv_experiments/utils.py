import torch
from torch import nn, optim
from models import CNNet, TCNet

def load_data(args):
    #TODO
    pass

def get_criterion(args):
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    return criterion

def get_optimizer(args, model):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = args.weight_decay)
    return optimizer

def get_model(args):
    if args.model == 'cnn':
        model = CNNet(args.n_series, args.pooling_strategy)
    elif args.model == 'tcn':
        model = TCNet()
    return model

def get_scheduler(args, optimizer):
    if args.scheduler is None:
        lr_scheduler = None
    elif args.scheduler == 'plateau':
        lr_scheduler = optim.ReduceLROnPlateau(optimizer, patience=args.patience)
    elif args.scheduler == 'multistep':
        lr_scheduler = optim.MultiStepLR(optimizer, milestones=args.milestones)
    return lr_scheduler
