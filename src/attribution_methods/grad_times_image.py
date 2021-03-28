import numpy as np

from src.Plot_tools import max_regarding_to_abs
from src.attribution_methods import vanilla_saliency


def generate_grad_times_image_saliency(model, image, target_class, device, make_single_channel=True, grad=None):
    if grad is None:
        vanilla = vanilla_saliency.VanillaSaliency(model, device)
        saliency = vanilla.generate_saliency(image, target_class, False)
    else:
        saliency = grad
    img = image.detach().cpu().numpy().squeeze(0)
    if not isinstance(saliency, np.ndarray):
        saliency = saliency.detach().cpu().numpy()

    saliency = np.asarray(saliency) * img
    if len(saliency.shape) == 4:
        saliency = saliency.squeeze(0)
    if make_single_channel:
        saliency = max_regarding_to_abs(np.max(saliency, axis=0), np.min(saliency, axis=0))
    return saliency