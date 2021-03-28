from copy import deepcopy

import numpy as np

from src import Plot_tools
from src.attribution_methods.vanilla_saliency import VanillaSaliency


def convert_to_grayscale(im_as_arr):
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


class IntegratedGradients():

    def __init__(self, model, device):
        self.model = model
        self.gradients = None
        self.device = device
        # Put model in evaluation mode
        self.model.eval()

    def generate_images_on_linear_path(self, input_image, steps):
        step_list = np.arange(steps + 1) / steps
        xbar_list = [input_image * step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, target_class, make_single_channel=True):
        vanillaSaliency = VanillaSaliency(self.model, self.device)
        saliency = vanillaSaliency.generate_saliency(input_image, target_class, make_single_channel)
        if not isinstance(saliency, np.ndarray):
            saliency = saliency.detach().cpu().numpy()
        return saliency

    def generate_integrated_gradients(self, input_image, target_class, steps, make_single_channel=True):
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        integrated_grads = None
        for xbar_image in xbar_list:
            single_integrated_grad = self.generate_gradients(xbar_image, target_class, False)
            if integrated_grads is None:
                integrated_grads = deepcopy(single_integrated_grad)
            else:
                integrated_grads = (integrated_grads + single_integrated_grad)
        integrated_grads /= steps
        saliency = integrated_grads[0]
        img = input_image.detach().cpu().numpy().squeeze(0)
        saliency = np.asarray(saliency) * img
        if make_single_channel:
            saliency = Plot_tools.max_regarding_to_abs(np.max(saliency, axis=0), np.min(saliency, axis=0))
        return saliency


def generate_integrad_saliency_maps(model, preprocessed_image, label, device, steps=100, make_single_channel=True):
    IG = IntegratedGradients(model, device)
    integrated_grads = IG.generate_integrated_gradients(preprocessed_image, label, steps, make_single_channel)
    if make_single_channel:
        integrated_grads = convert_to_grayscale(integrated_grads)
    return integrated_grads