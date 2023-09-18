import torch
import torch.nn as nn
from torchsummary import summary
# load rnn model


'''https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/'''


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_first = True) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = batch_first) # intialize rnn
        self.fc = nn.Linear(hidden_size, output_size) # initialize fully connected layer
        
        
    def forward(self, x):
        if self.batch_first:
            batch_size = x.size(0)
        else:
            batch_size = x.size(1)
        y = self.rnn(x) # returns output and hidden state as tuple
        o = self.fc(y[0]) # get output from hidden state
        return o, y[1] # return output and hidden state
        


rnn = RNN(input_size=10, hidden_size=20, num_layers=2, batch_first = True).to('cuda')
x = torch.randn(50, 32, 10).to('cuda')
print(rnn.rnn.forward(x)[0].shape) 