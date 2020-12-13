import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

npy_filename = 'processed_endomondoHR_proper_interpolate.npy'
temporal_pickle_filename = 'endomondoHR_proper_temporal_dataset.pkl'
metadata_pickle_filename = 'endomondoHR_proper_metaData.pkl'

class TSData(Dataset):
    def __init__(self, x, y):
        super(TSData, self).__init__()
        self.x = x
        self.y = y
        
        assert self.x.shape[0] == self.y.shape[0]
        assert self.x.shape[-1] == self.y.shape[-1]
    
    def __getitem__(self, i):
        n = self.x.shape[2]
        return self.x[i,:,:n-1], self.y[i,:,1:]
    
    def __len__(self):
        return self.x.shape[0]

class ClassData(Dataset):
    def __init__(self, x, y):
        super(TSData, self).__init__()
        self.x = x
        self.y = y
        
        assert self.x.shape[0] == self.y.shape[0]
    
    def __getitem__(self, i):
        n = self.x.shape[2]
        return self.x[i,:,:n-1], self.y[i]
    
    def __len__(self):
        return self.x.shape[0]

def map_data(train_idx, val_idx, test_idx, context, npy_data):
    idxMap = {}
    for idx, d in enumerate(npy_data):
        idxMap[d['id']] = idx
    
    train_idx = set.intersection(set(train_idx), set(idxMap.keys()))
    val_idx = set.intersection(set(val_idx), set(idxMap.keys()))
    test_idx = set.intersection(set(test_idx), set(idxMap.keys()))
    
    train_idx = [idxMap[wid] for wid in train_idx]
    val_idx = [idxMap[wid] for wid in val_idx]
    test_idx = [idxMap[wid] for wid in test_idx]
    
    return train_idx, val_idx, test_idx
    
def get_class_data(args, y):
    if 'sport' in args.y_vals:
        indices = np.where(np.logical_or.reduce((np.core.defchararray.find(y.flatten(),'bike')!=-1,np.core.defchararray.find(y.flatten(),'cycling')!=-1, np.core.defchararray.find(y.flatten(),'run')!=-1)))[0]
        y = y[indices]
        int_y = np.array([1 if 'run' in e else 0 for e in y.flatten()])
    return indices, int_y

def get_npy_data(args):
    raw_data = np.load(f'{args.dataset_path}/{npy_filename}', allow_pickle=True)[0]
    input = np.array(list(map(lambda x: [x[col] for col in args.x_vals], raw_data)))
    y = np.array(list(map(lambda x: [x[col] for col in args.y_vals], raw_data)))
    
    tensor_input = torch.from_numpy(input).float()
    
    if args.task == 'forecasting':
        tensor_y = torch.from_numpy(y).float()
        
    elif args.task == 'prediction':
        indices, int_y = get_class_data(args, y)
        tensor_input = tensor_input[indices]
        tensor_y = torch.from_numpy(int_y)
        
    return raw_data, tensor_input, tensor_y

def get_metadata(args):
    with open(f'{args.dataset_path}/{temporal_pickle_filename}', 'rb') as f:
        train_idx, val_idx, test_idx, context = pickle.load(f)
    
    return train_idx, val_idx, test_idx, context
