import numpy as np
from torch.autograd import Variable

from src.Plot_tools import max_regarding_to_abs


class VanillaSaliency():

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def generate_saliency(self, data, label, make_single_channel=True):
        data_var_sal = Variable(data).to(self.device)
        self.model.zero_grad()
        if data_var_sal.grad is not None:
            data_var_sal.grad.data.zero_()
        data_var_sal.requires_grad_(True)

        out_sal = self.model(data_var_sal)
        out_sal[0, label].backward(retain_graph=True)
        grad = data_var_sal.grad.data.detach().cpu()
        if make_single_channel:
            grad = np.asarray(grad.detach().cpu().squeeze(0))
            grad = max_regarding_to_abs(np.max(grad, axis=0), np.min(grad, axis=0))
            grad = np.expand_dims(grad, axis=0)
        else:
            grad = np.asarray(grad)
        return grad
