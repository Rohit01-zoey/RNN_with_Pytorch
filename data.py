''' All the data that is loaded should be (B, N, L) for batched data, where B is the batch size, N is the number of channels, and L is the length of the signal. '''


import torch
import torch.nn as nn  
from torch.utils.data import random_split, Dataset 

class CustomDataLoader(Dataset):
    def __init__(self, data,  labels = None):
        self.data = data
        if labels not in [None, []]:
            self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if hasattr(self, 'labels'):
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]

def get_dataloader(data, batch_size = 32, shuffle = True):
    train_dataset, test_data = random_split(data, [int(0.8*len(data)), len(data) - int(0.8*len(data))])
    train_dataset, val_dataset = random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset) - int(0.8*len(train_dataset))])
    
    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle)
    val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = shuffle)
    test_data = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = shuffle)
    return train_dataset, val_dataset, test_data

# cc = CustomDataLoader(torch.randn(500, 200, 10))
# print(len(cc))
# cc_tr, cc_val = torch.utils.data.random_split(cc, [0.8, 0.2])
# print(len(cc_tr))