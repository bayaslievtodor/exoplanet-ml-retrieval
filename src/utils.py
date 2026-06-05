import random
import numpy as np
import torch


def set_seeds(seed=123):
    #fix 123 seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def pinball_loss(preds, targets):

    # boilerplate pinball loss math 

    # Define target percentiles matching your data structure
    quantiles = torch.tensor([0.16, 0.50, 0.84], device=preds.device)
    # Tile to match the 15-wide output vector
    quantiles = quantiles.repeat(5) 
    
    error = targets - preds
    # The pinball formula: max(q * error, (q - 1) * error)
    loss = torch.max(quantiles * error, (quantiles - 1) * error)
    return torch.mean(loss)