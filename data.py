''' All the data that is loaded should be (N, L) for unbatched data and (B, N, L) for batched data, where B is the batch size, N is the number of channels, and L is the length of the signal. '''


import torch
import torch.nn as nn   

class CustomDataLoader(torch.utils.data.Dataset):
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

def get_dataloader(data):
    return torch.utils.data.DataLoader(data, batch_size = 32, shuffle = True)

cc = CustomDataLoader(torch.randn(100, 10), torch.randn(100, 1))
print(hasattr(cc, 'labels'))