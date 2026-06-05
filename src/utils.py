import random
import numpy as np
import torch


def set_seeds(seed=123):
    #fix 123 seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def pinball_loss(preds, targets):

    quantiles = torch.tensor([0.16, 0.50, 0.84], device=preds.device) # iintervals
    quantiles = quantiles.repeat(5) 
    
    error = targets - preds
    loss = torch.max(quantiles * error, (quantiles - 1) * error)
    return torch.mean(loss)