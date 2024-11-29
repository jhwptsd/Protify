import numpy as np
import torch

def RMSD(p1, p2):
    loss = torch.sqrt(torch.mean(p1[:len(p2)], p2)**2)
    return loss

def tm_score(p1, p2, lt):
    d0 = lambda l: 1.24 * np.cbrt(l-15) - 1.8
    loss = torch.mean(1/(1+np.power(np.abs(np.linalg.norm(p1-p2))/d0(lt),2)))
    return loss