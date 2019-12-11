import numpy as np
import tinyobjloader
import torch

def normalize(data, axis):
    return data / torch.norm(data, p=None, dim=axis, keepdim=True)
