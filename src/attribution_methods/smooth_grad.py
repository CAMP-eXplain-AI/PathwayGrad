import numpy as np
import torch
from torch.autograd import Variable


def generate_smooth_grad(Backprop, prep_img, target_class, param_n, param_sigma_multiplier, single_channel=True):
    smooth_grad = None

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(prep_img) - torch.min(prep_img)).item()
    for x in range(param_n):
        noise = Variable(prep_img.data.new(prep_img.size()).normal_(mean, sigma ** 2))
        noisy_img = prep_img + noise
        vanilla_grads = Backprop.generate_saliency(noisy_img, target_class, single_channel)
        if not isinstance(vanilla_grads, np.ndarray):
            vanilla_grads = vanilla_grads.detach().cpu().numpy()
        if smooth_grad is None:
            smooth_grad = vanilla_grads
        else:
            smooth_grad = smooth_grad + vanilla_grads

    smooth_grad = smooth_grad / param_n
    return torch.from_numpy(smooth_grad)