import copy

import matplotlib.cm as mpl_color_map
import numpy as np
import torch
import torchvision
from PIL import Image

from src.models.classification.ConvNetSimple import ConvNetSimple
from src.models.classification.PytorchCifarResnet import ResNet as CifarResnet


def apply_colormap_on_image(org_im, activation, colormap_name):
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))
    if type(org_im) == np.ndarray:
        org_im = Image.fromarray(np.uint8(org_im * 255), 'RGB')
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


class CamExtractor():

    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.device = device
        self.backward_handle = None

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
            relu5_2 is penultimate layer for VGG-11 - module_pos = 19 in pytorch
            relu5_2 is penultimate layer for VGG-13 - module_pos = 23 in pytorch
            relu5_3 for VGG-16 - Check original paper Page 8, first paragraph, module_pos = 29 in pytorch
            relu5_4 for VGG-19 - https://github.com/ramprs/grad-cam,  module_pos = 35 in pytorch
            relu5 for AlexNet - https://github.com/ramprs/grad-cam
            'layer4' for resnet
        """
        conv_output = None

        def save_gradient(grad_out):
            # import pydevd
            # pydevd.settrace(suspend=False, trace_only_current_thread=True)
            self.gradients = grad_out

        if type(self.model) == torchvision.models.vgg.VGG:
            for module_pos, module in self.model.features._modules.items():
                x = module(x)  # Forward
                if int(module_pos) == self.target_layer:
                    # print(f'GradCam working on {module_pos}, {module}')
                    self.backward_handle = x.register_hook(save_gradient)
                    conv_output = x  # Save the convolution output on that layer
        elif type(self.model) in [torchvision.models.resnet.ResNet, CifarResnet]:
            def hook_feature(module, input, output):
                self.features = output.clone().detach()

            def hook_gradient(module, grad_in, grad_out):
                # import pydevd
                # pydevd.settrace(suspend=False, trace_only_current_thread=True)
                self.gradients = grad_out[0].clone().detach()

            c = list(self.model.children())[
                self.target_layer[0]]  # Get Last Sequential Block, before AdaptiveAvgPool2d and Linear layer
            relu_layer = c[self.target_layer[1]]
            handle = relu_layer.register_forward_hook(hook_feature)
            self.backward_handle = relu_layer.register_backward_hook(hook_gradient)
            x = self.model(x)
            handle.remove()
            conv_output = self.features
        elif type(self.model) == ConvNetSimple:
            def hook_feature(module, input, output):
                self.features = output.clone().detach()

            def hook_gradient(module, grad_in, grad_out):
                # import pydevd
                # pydevd.settrace(suspend=False, trace_only_current_thread=True)
                self.gradients = grad_out[0].clone().detach()

            # Get Last Convolutional layer relu
            relu_layer = list(self.model.children())[self.target_layer]
            handle = relu_layer.register_forward_hook(hook_feature)
            self.backward_handle = relu_layer.register_backward_hook(hook_gradient)
            x = self.model(x)
            handle.remove()
            conv_output = self.features

        return conv_output, x

    def forward_pass(self, x):
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten

        if type(self.model) == torchvision.models.vgg.VGG:
            x = self.model.classifier(x)
        return conv_output, x


class GradCam():

    def __init__(self, model, target_layer, device):
        self.model = model
        self.model.eval()
        self.device = device
        self.extractor = CamExtractor(self.model, target_layer, self.device)

    def generate_cam(self, input_image, target_class=None):
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        one_hot_output = one_hot_output.to(self.device)
        self.model.zero_grad()
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        guided_gradients = self.extractor.gradients.cpu().data.numpy()[0]
        self.extractor.backward_handle.remove()
        target = conv_output.cpu().data.numpy()[0]
        weights = np.mean(guided_gradients, axis=(1, 2))
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                    input_image.shape[3]), Image.ANTIALIAS)) / 255
        return cam


def generate_grad_cam_saliency_maps(model, original_image, preprocessed_image, label, target_layer, device):
    grad_cam = GradCam(model, target_layer=target_layer, device=device)
    cam = grad_cam.generate_cam(preprocessed_image, label)

    # Apply Grayscale activation map if original image is provided
    heatmap, heatmap_on_image = None, None
    if original_image is not None:
        heatmap, heatmap_on_image = apply_colormap_on_image(original_image, cam, 'hsv')

    # Return colored heatmap, heatmap on image, grayscale heatmap
    return heatmap, heatmap_on_image, cam


def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb