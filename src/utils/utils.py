import datetime
import os
import random

import gc
import numpy as np
import torch


def set_random_seed(random_seed):
    """
    Set seed for random seed generator for python, pytorch and numpy.
    :param random_seed: Initial seed value
    :return: None
    """
    torch.manual_seed(random_seed)  # Set for pytorch, used for cuda as well.
    random.seed(random_seed)        # Set for python
    np.random.seed(random_seed)     # Set for numpy


def make_results_dir(outdir='./logs/'):
    """
    Create a timestamped output directory
    :return: The output directory path.
    """
    timestamp = datetime.datetime.now().isoformat()
    dirname = f'{timestamp}'
    outdir = os.path.join(os.getcwd(), outdir, dirname)
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}")
    return outdir


def monitor_gpu_memory():
    # To keep check of memory
    print("Number of objects ", len(gc.get_objects()))
    count = 0
    for obj in sorted(gc.get_objects(), key=lambda x: id(x)):
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(id(obj), type(obj), obj.size())
                count+=1
        except Exception as e:
            pass
    print("Total GPU memory in use: ", torch.cuda.memory_allocated(device=None))
    print("Number of tensors ", count)
