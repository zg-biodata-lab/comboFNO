import re
import random
import torch
import numpy as np

def sanitize_filename(filename):
    """
    Clean file name: Remove illegal characters, replace spaces
    filename: Original file name
    return: Cleaned security file name
    """
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = sanitized.replace(' ', '_')
    return sanitized

def set_seed(seed):
    """
    Set global random seeds
    seed: random seed value
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False