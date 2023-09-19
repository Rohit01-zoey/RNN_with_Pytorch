import torch
from torchsummary import summary


def get_device():
    """Returns the device to be used for tensor operations.

    Returns:
        torch.device: the device to be used for tensor operations
    """
    if torch.cuda.is_available():
        print('Using GPU....')
        return torch.device('cuda')
    else:
        print("Defaulting to CPU....")
        return torch.device('cpu')
    
    
def load_model(path : str):
    """Loads the model from the saved checkpoint file.

    Returns:
        torch.nn.Module: the model loaded from the checkpoint file
    """
    try:
        model = torch.load(path)
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {str(e)}")
        return None 


def save_model(model : torch.nn.Module, path : str):
    """Saves the model to the specified path.

    Args:
        model (torch.nn.Module): the model to be saved
        path (str): the path to save the model to
    """
    try:
        torch.save(model, path)
    except Exception as e:
        print(f"An error occurred while saving the model: {str(e)}")
        
        
def get_model_summary(model : torch.nn.Module, input_size : tuple):
    """Prints the model summary.

    Args:
        model (torch.nn.Module): the model whose summary is to be printed
        input_size (tuple): the size of the input tensor
    """
    try:
        summary(model, input_size)
    except Exception as e:
        print(f"An error occurred while getting the model summary: {str(e)}")
        

def train_with_label(model, data, loss, optimizer, epochs, device, batch_size):
    """Trains the model. Assumes that the data is labelled.

    Args:
        model (torch.nn.Module): the model to be trained
        data (torch.Tensor): the data to be used for training
        loss (torch.nn.Module): the loss function to be used
        optimizer (torch.optim.Optimizer): the optimizer to be used for training
    """
    for iteration in range(epochs):
        for batch in range(0, data.size(0), batch_size):
            x, y = data[batch:batch + batch_size, :-1, :], data[batch:batch + batch_size, -1, :]
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            loss_val = loss(out, y)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            print(f'Epoch: {iteration + 1}, Loss: {loss_val.item()}')