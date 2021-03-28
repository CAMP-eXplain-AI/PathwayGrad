import numpy as np
import torch
from torch.nn import ReLU

from src.attribution_methods.vanilla_saliency import VanillaSaliency


class GuidedBackprop():

    def __init__(self, model, device):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.device = device
        self.hooks = []
        self.model.eval()
        self.update_relus()

    def update_relus(self):

        def relu_backward_hook_function(module, grad_in, grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for module in self.model.modules():
            if isinstance(module, ReLU):
                self.hooks.append(module.register_backward_hook(relu_backward_hook_function))
                self.hooks.append(module.register_forward_hook(relu_forward_hook_function))

    def generate_gradients(self, input_image, target_class, make_single_channel=True):
        vanillaSaliency = VanillaSaliency(self.model, self.device)
        sal = vanillaSaliency.generate_saliency(input_image, target_class, make_single_channel)
        if not isinstance(sal, np.ndarray):
            sal = sal.detach().cpu().numpy()
        for hook in self.hooks:
            hook.remove()
        return sal