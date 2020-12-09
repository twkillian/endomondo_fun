import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

npy_filename = 'processed_endomondoHR_proper_interpolate.npy'
temporal_pickle_filename = 'endomondoHR_proper_temporal_dataset.pkl'
metadata_pickle_filename = 'endomondoHR_proper_metaData.pkl'

class TSData(Dataset):
    def __init__(self, ts):
        super(TSData, self).__init__()
        self.x = ts
    
    def __getitem__(self, i):
        n = self.x.shape[2]
        return self.x[i,:,:n-1], self.x[i,:,1:]
    
    def __len__(self):
        return len(self.x)

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

def get_npy_data(args):
    raw_data = np.load(f'{args.dataset_path}/{npy_filename}', allow_pickle=True)[0]
    if args.n_series == 2:
        np_ts_data = np.array(list(map(lambda x: [x['tar_heart_rate'], x['tar_derived_speed']], raw_data)))
    elif args.n_series == 1:
        np_ts_data = np.array(list(map(lambda x: [x[args.y_val]], raw_data)))
    else:
        raise ValueError('Invalid number of time series specified')
    
    tensor_ts_data = torch.from_numpy(np_ts_data).float()
    return raw_data, tensor_ts_data

def get_metadata(args):
    with open(f'{args.dataset_path}/{temporal_pickle_filename}', 'rb') as f:
        train_idx, val_idx, test_idx, context = pickle.load(f)
    
    return train_idx, val_idx, test_idx, context
