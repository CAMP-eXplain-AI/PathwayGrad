import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append('./')
from cdrp import main as cdrp
from cdrp import main2 as r_cdrp
from src import Plot_tools


class Pruner:
    def __init__(self, model, input, device, label = None, output_orig=None):
        self.model = model
        self.input = input
        self.model.eval()
        self.device = device
        # pruned_activations_mask, 0 for pruned, 1 for others
        self.gradients = []
        self.activations = []
        self.pruned_activations_mask = []
        if label and output_orig is not None:
            self.label = label
            self.output_orig = output_orig
        else:
            self.output_orig = self.model(input).detach()
            self.label = self.output_orig.data.max(1)[1].item()
        self.handles_list = []
        self._hook_layers()
        self.integrad_handles_list = []
        self.integrad_scores = []
        self.integrad_calc_activations_mask = None

    def reset(self):
        self.remove_handles()
        self.model.eval()
        self.gradients = []
        self.activations = []
        self.pruned_activations_mask = []
        self.handles_list = []
        self._hook_layers()
        self.integrad_handles_list = []
        self.integrad_scores = []
        self.integrad_calc_activations_mask = None

    def _hook_layers(self):
        def backward_hook_relu(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].to(self.device))

        def forward_hook_relu(module, input, output):
            # mask output by pruned_activations_mask
            # In the first model(input) call, the pruned_activations_mask
            # is not yet defined, thus we check for emptiness
            if self.pruned_activations_mask:
                if len(self.activations) >= 15:
                    self.activations = self.activations[15:]
                    self.gradients = self.gradients[15:]
                output = torch.mul(output, self.pruned_activations_mask[len(self.activations)].to(self.device)) #+ self.pruning_biases[len(self.activations)].to(self.device)
            self.activations.append(output.to(self.device))
            return output

        i = 0
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
            # if isinstance(module, resnet.BasicBlock):
                self.handles_list.append(module.register_forward_hook(forward_hook_relu))
                self.handles_list.append(module.register_backward_hook(backward_hook_relu))

    def remove_handles(self):
        for handle in self.handles_list:
            handle.remove()
        self.handles_list.clear()
        self.activations = []
        self.gradients = []

    # we always call _forward instead of directly calling model(input),
    # since the activations and gradients need to be reset
    def _forward(self, input):
        self.activations = []
        self.gradients = []
        self.model.zero_grad()
        output = self.model(input)
        return output

    def _initialize_pruned_mask(self):
        output = self._forward(self.input)

        # initializing pruned_activations_mask
        for layer in self.activations:
            self.pruned_activations_mask.append(torch.ones(layer.size()).to(self.device))
        return output

    def _number_of_neurons(self):
        total = 0
        if self.pruned_activations_mask:
          for layer in self.pruned_activations_mask:
              num_neurons_in_layer = layer.numel()
              total += num_neurons_in_layer
        else:
          for layer in self.activations:
              num_neurons_in_layer = layer.numel()
              total += num_neurons_in_layer
        return total

    def _number_of_pruned_neurons(self):
        total = 0
        total_kept = 0

        for mask in self.pruned_activations_mask:
            num_neurons_in_layer = mask.numel()
            num_of_kept_neurons_in_layer = mask.sum()
            total += num_neurons_in_layer
            total_kept += num_of_kept_neurons_in_layer
        return (total - total_kept), total_kept

    def _compute_taylor_scores(self):

        first_order_taylor_scores = []
        self.gradients.reverse()

        for i, layer in enumerate(self.activations):
            first_order_taylor_scores.append(torch.abs(torch.mul(layer, self.gradients[i])))

        return first_order_taylor_scores

    def _mask_least_important_neurons(self, first_order_taylor_scores, percentile_to_prune, debug=False):

        scores_all_layers = np.empty(0)
        for layer_scores in first_order_taylor_scores:
            scores_all_layers = np.concatenate((scores_all_layers,
                                                layer_scores.cpu().detach().numpy().flatten()))
        
        remove_threshold = np.percentile(scores_all_layers, percentile_to_prune)
        copy_first_order_taylor_scores = first_order_taylor_scores.copy()

        for i, layer_scores in enumerate(first_order_taylor_scores):
            if debug:
                print("top neuron info:", "layer's shape", layer_scores.cpu().detach().numpy().shape, "layer num:", i, ", #####", "index of top neuron", np.argmax(layer_scores.cpu().detach().numpy().flatten()))
            self.pruned_activations_mask[i][layer_scores <= remove_threshold] = 0
            copy_first_order_taylor_scores[i][layer_scores <= remove_threshold] = 0

        return copy_first_order_taylor_scores

    def _mask_least_important_neurons_iterative(self, first_order_taylor_scores, percentile_to_prune):

        scores_all_layers = np.empty(0)
        for layer_scores in first_order_taylor_scores:
            scores_all_layers = np.concatenate((scores_all_layers,
                                                layer_scores.cpu().detach().numpy().flatten()))

        scores_all_layers[scores_all_layers <= 0] = np.max(scores_all_layers.flatten())
        remove_threshold = np.percentile(scores_all_layers, percentile_to_prune)
        copy_first_order_taylor_scores = first_order_taylor_scores.copy()

        for i, layer_scores in enumerate(first_order_taylor_scores):
            self.pruned_activations_mask[i][layer_scores <= remove_threshold] = 0
            self.pruned_activations_mask[i][layer_scores <= 0] = 1


    def prune_neuron_mct(self, percentile_to_prune=85., debug=False):

        initial_output = self._initialize_pruned_mask()
        initial_output = torch.nn.functional.softmax(initial_output, dim=1)
        initial_predicted_logit = initial_output.data.max(1)[0].item()
        initial_predicted_class = initial_output.data.max(1)[1].item()
        if debug:
            print("Initial output = {}".format(initial_predicted_logit))
            print('Initial predicted class {}: '.format(initial_predicted_class))
        label = torch \
            .tensor([self.label]).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        initial_loss = criterion(initial_output, label)

        num_total = self._number_of_neurons()
        if debug:
            print('initial loss {}'.format(initial_loss))
            print("total number of neurons: {}".format(num_total))

        output = self._forward(self.input)
        output[0, self.label].backward(retain_graph=True)
        first_order_taylor_scores = self._compute_taylor_scores()
        self._mask_least_important_neurons(first_order_taylor_scores, percentile_to_prune, debug)
        output = self._forward(self.input)
        output_softmax = torch.nn.functional.softmax(output, dim=1)
        output_orig_softmax = torch.nn.functional.softmax(self.output_orig, dim=1)
        return abs(output_softmax[0, self.label].data.item() - output_orig_softmax[0, self.label].data.item()) / output_orig_softmax[0, self.label].data.item()

    def prune_random(self, percentile_to_prune, debug=False):
        initial_output = self._initialize_pruned_mask()
        initial_output = torch.nn.functional.softmax(initial_output, dim=1)
        initial_predicted_logit = initial_output.data.max(1)[0].item()
        initial_predicted_class = initial_output.data.max(1)[1].item()
        if debug:
            print("Initial output = {}".format(initial_predicted_logit))
            print('Initial predicted class {}: '.format(initial_predicted_class))
        label = torch \
            .tensor([self.label]).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        initial_loss = criterion(initial_output, label)

        num_total = self._number_of_neurons()
        if debug:
            print('initial loss {}'.format(initial_loss))
            print("total number of neurons: {}".format(num_total))

        output = self._forward(self.input)
        output[0, self.label].backward(retain_graph=True)

        for i, layer_scores in enumerate(self.activations):
            #print(layer_scores.shape, len(layer_scores.shape))
            size_of_layer = layer_scores.view(-1).shape[0]
            mask = torch.ones(size_of_layer)
            mask[np.random.permutation(range(0, size_of_layer))[:int(percentile_to_prune/100*size_of_layer)]] = 0

            if len(layer_scores.shape) == 2:
               mask = mask.view(layer_scores.shape[0], layer_scores.shape[1])

            elif len(layer_scores.shape) == 3:
               mask = mask.view(layer_scores.shape[0], layer_scores.shape[1], layer_scores.shape[2])

            elif len(layer_scores.shape) == 4:
               mask = mask.view(layer_scores.shape[0], layer_scores.shape[1], layer_scores.shape[2], layer_scores.shape[3])

            self.pruned_activations_mask[i] = mask

    def _init_integrad_mask(self):
        self.integrad_calc_activations_mask = []
        _ = self._forward(self.input)
        for a in self.activations:
            self.integrad_calc_activations_mask.append(torch.ones(a.shape))


    def _calc_integrad_scores(self, iterations):
        def forward_hook_relu(module, input, output):
            output = torch.mul(output, self.integrad_calc_activations_mask[len(self.activations)-1].to(self.device))
            return output

        initial_output = self._initialize_pruned_mask()
        output = self._forward(self.input)
        output[0, self.label].backward(retain_graph=True)

        original_activations = []
        for a in self.activations:
            original_activations.append(a.detach().clone())

        self._init_integrad_mask()
        mask_step = 1./iterations
        i = 0
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
               self.integrad_scores.append(torch.zeros(original_activations[i].shape).to(self.device))
               self.integrad_calc_activations_mask[i] = torch.zeros(self.integrad_calc_activations_mask[i].shape)
               self.integrad_handles_list.append(module.register_forward_hook(forward_hook_relu))

               for j in range(iterations+1):
                   self.integrad_calc_activations_mask[i] += j*mask_step
                   output = self._forward(self.input)
                   output[0, self.label].backward(retain_graph=True)
                   self.gradients.reverse()
                   self.integrad_scores[len(self.integrad_scores)-1] += self.gradients[i]
               self.integrad_scores[len(self.integrad_scores)-1] = abs(self.integrad_scores[len(self.integrad_scores)-1]/(iterations+1) * original_activations[i])
               self.integrad_calc_activations_mask[i] = torch.ones(self.integrad_calc_activations_mask[i].shape)
               self.integrad_handles_list[0].remove()
               self.integrad_handles_list.clear()
               i += 1

    def prune_integrad(self, percentile_to_prune, iterations=10, debug=False):
        self._calc_integrad_scores(iterations)
        scores_all_layers = np.empty(0)
        for layer_scores in self.integrad_scores:
            scores_all_layers = np.concatenate((scores_all_layers,
                                                layer_scores.cpu().detach().numpy().flatten()))

        remove_threshold = np.percentile(scores_all_layers, percentile_to_prune)
        copy_integrad_scores = self.integrad_scores.copy()

        #for l in self.integrad_scores:
        #    print(np.argmax(l.cpu().detach().numpy()))

        for i, layer_scores in enumerate(self.integrad_scores):

            self.pruned_activations_mask[i][layer_scores <= remove_threshold] = 0
            copy_integrad_scores[i][layer_scores <= remove_threshold] = -1

        return copy_integrad_scores


    def prune_greedy(self, percentile_to_prune=1., iteration=100, debug=False):

        initial_output = self._initialize_pruned_mask()
        initial_output = torch.nn.functional.softmax(initial_output, dim=1)
        initial_predicted_logit = initial_output.data.max(1)[0].item()
        initial_predicted_class = initial_output.data.max(1)[1].item()
        if debug:
            print("Initial output = {}".format(initial_predicted_logit))
            print('Initial predicted class {}: '.format(initial_predicted_class))
        label = torch \
            .tensor([self.label]).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        initial_loss = criterion(initial_output, label)
        if debug:
            print('initial loss {}'.format(initial_loss))

        num_total = self._number_of_neurons()
        if debug:
            print("total number of neurons: {}".format(num_total))

        output = self._forward(self.input)
        output[0, self.label].backward(retain_graph=True)

        first_order_taylor_scores = self._compute_taylor_scores()
        for i in range(iteration):
            #print("iteration: {}".format(i))
            self._mask_least_important_neurons_iterative(first_order_taylor_scores, i*percentile_to_prune)
            output = self._forward(self.input)
            output[0, self.label].backward(retain_graph=True)
            first_order_taylor_scores = self._compute_taylor_scores()

    def prune_dgr(self, percentile_to_prune, r=False, debug=False):
        _ = self.model(self.input)
        acts = []
        for a in self.activations:
            acts.append(a.detach().clone())
        self.remove_handles()
        if not r:
            paths = cdrp.get_path(self.model, self.input, self.label, percentile_to_prune, acts)
        else:
            paths = r_cdrp.get_path(self.model, self.input, self.label, percentile_to_prune, acts)

        scores_all_layers = np.empty(0)
        for path_scores in paths:
            if debug:
                print("top neuron info:", "layer's shape", path_scores.cpu().detach().numpy().shape, "layer num:", ", #####", "index of top neuron", np.argmax(path_scores.cpu().detach().numpy().flatten()))          
            scores_all_layers = np.concatenate((scores_all_layers,
                                                path_scores.cpu().detach().numpy().flatten()))

        remove_threshold = np.percentile(scores_all_layers, percentile_to_prune)
        self._hook_layers()
        _ = self._forward(self.input)

        initial_output = self._initialize_pruned_mask()
        for i, p in enumerate(paths):
            self.pruned_activations_mask[i][p <= remove_threshold] = 0
            self.pruned_activations_mask[i][p > remove_threshold] = 1
			
    def base_sparsity(self):
        num = 0
        #n = 0
        self._forward(self.input)
        for activation in self.activations:
            activation_copy = activation.clone().detach()
            #print(activation_copy.view(-1)[2])
            activation_copy[activation_copy <= 0.] = -1
            activation_copy[activation_copy > 0.] = 0
            activation_copy = abs(activation_copy)
            #print(activation_copy.view(-1)[2])
            num += torch.norm(activation_copy.view(-1), p=1).item()
            #num += sum(activation_copy.view(-1))
            #print(num)
            #n += activation_copy.view(-1).shape[0]
        num_total = self._number_of_neurons()
        # print(num_total)
        return num/num_total

    def dead_neurons_path(self):
        initial_output = self._initialize_pruned_mask()
        self._forward(self.input)
        for i, activation in enumerate(self.activations):
            activation_copy = activation.clone().detach()
            activation_copy[activation_copy <= 0.] = -1
            activation_copy[activation_copy > 0.] = 0
            activation_copy = abs(activation_copy)
            self.pruned_activations_mask[i] = activation_copy

    def generate_saliency(self, make_single_channel=True):
        data_var_sal = Variable(self.input).to(self.device)

        if data_var_sal.grad is not None:
            data_var_sal.grad.data.zero_()
        data_var_sal.requires_grad_(True)

        out_sal = self._forward(data_var_sal)
        out_sal[0, self.label].backward(retain_graph=True)

        grad = data_var_sal.grad.data.detach().cpu()
        grad_np = np.asarray(grad).squeeze(0)

        if make_single_channel:
            grad = Plot_tools.max_regarding_to_abs(np.max(grad_np, axis=0), np.min(grad_np, axis=0))
            return np.expand_dims(grad, axis=0)
        # The output is 3-channel image size saliency map
        return np.expand_dims(grad_np, axis=0)
