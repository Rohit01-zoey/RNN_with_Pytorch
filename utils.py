import torch
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
