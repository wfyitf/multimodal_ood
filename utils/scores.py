import pandas as pd
import numpy as np
import torch.nn.functional as F

def cosine_similarity_torch(x, y):
    return F.cosine_similarity(x, y, dim=1).item()

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
