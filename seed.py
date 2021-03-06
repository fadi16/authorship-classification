import random
import numpy as np
import torch

# set a random seed for reproducibility
def seed_for_reproducability(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True