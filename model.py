import torch
import torch.nn as nn
from torchsummary import summary
# load rnn model


'''https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/'''


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device) -> None:
        super().__init__()
        self.hidden_size = hidden_size # hidden size of rnn
        self.num_layers = num_layers # number of layers in rnn
        self.device = device # device to be used for tensor operations
        
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True) # intialize rnn
        self.fc = nn.Linear(hidden_size, output_size) # initialize fully connected layer
        
        
    def forward(self, x):
        batch_size = x.size(0) # getting the batch size
        hidden = self._init_hidden(batch_size).to(self.device) # rnn expects hidden state as input along with input tensor
        out, hidden = self.rnn(x, hidden) # returns output and hidden state as tuple
        # out = out.contiguous().view(-1, self.hidden_size) # reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out) # get output from hidden state
        return out, hidden # return output and hidden state
    
    def _init_hidden(self, batch_size, init_type = 'zeros'):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        if init_type == 'zeros':
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        elif init_type == 'normal':
            hidden = torch.randn(self.num_layers, batch_size, self.hidden_size)
        else:
            raise ValueError('Invalid initialization type')
        return hidden
        


# rnn = RNN(input_size=10, hidden_size=20, num_layers=2, output_size=10, device='cuda').to('cuda')
# x = torch.randn(32, 10).to('cuda')
# print(rnn.forward(x)[0].shape) 