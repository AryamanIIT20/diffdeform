import torch

def rescale(tensor, new_min = 0, new_max = 1):
    tensor = tensor.to(torch.float32)
    old_min = torch.min(tensor)
    old_max = torch.max(tensor)
    
    # Rescale to the [0, 1] range first
    scaled_tensor = (tensor - old_min) / (old_max - old_min)
    
    # Then scale to the [new_min, new_max] range
    rescaled_tensor = scaled_tensor * (new_max - new_min) + new_min
    
    return rescaled_tensor