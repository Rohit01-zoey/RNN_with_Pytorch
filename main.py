from chemotaxis_data import ChemotaxisDataLoader
from data import get_dataloader, CustomDataLoader
from model import RNN

import torch.nn as nn
import torch

cc = ChemotaxisDataLoader()
cc.shorten(500)
data = CustomDataLoader(cc.shortened_dataset, cc.shortened_dataset_labels)
train_dataset, val_dataset, test_data = get_dataloader(data, batch_size = 128, shuffle = True)
# print("Train dataset length: ", len(train_dataset))
# print("Validation dataset length: ", len(val_dataset))
# print("Test dataset length: ", len(test_data))

model = RNN(input_size=2, hidden_size=500, num_layers=10, output_size=2, device='cuda').to('cuda')

lr = 0.00001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(1000):
    loss = 0.0
    loss_val = 0.0
    for batch in train_dataset:
        optimizer.zero_grad()
        data, labels = batch  
        labels = labels.to('cuda') 
        # labels = labels.view(-1, 2)
        data = data.to('cuda')
        out, _ = model.forward(data)
        loss += criterion(out[:, :-1, :], labels[: , 1:, :])
    loss.backward()
    optimizer.step()
    
    
    for batch in val_dataset:
        data, labels = batch  
        labels = labels.to('cuda') 
        # labels = labels.view(-1, 2)
        data = data.to('cuda')
        out, _ = model.forward(data)
        loss_val += criterion(out[:, :-1, :], labels[: , 1:, :])
        
    print('Epoch: ', epoch, 'Train Loss: ', loss.item(), 'Val Loss: ', loss_val.item())
    
# prediction
import matplotlib.pyplot as plt
for batch in test_data:
    data, labels = batch  
    labels = labels.to('cuda') 
    # labels = labels.view(-1, 2)
    data = data.to('cuda')
    out, _ = model.forward(data)
    plt.figure()
    plt.plot(out[0:200, 0, 0].detach().cpu().numpy(), out[0:200, 0, 1].detach().cpu().numpy(), 'r.', label = 'Predicted')
    plt.plot(labels[0:200, 0, 0].detach().cpu().numpy(), labels[0:200, 0, 1].detach().cpu().numpy(), 'g.', label = 'Actual')
    plt.legend()
    plt.show()
    break
    
    
    
