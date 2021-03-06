# -*- coding: utf-8 -*-
"""Path Decoding.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1awxUcbGOCBN_VRTFVz3TtpLi7CTW300e

## Code to reproduce results of "Path Decoding" experiments

Paper: Khakzar, et al. "Neural Response Interpretation through the Lens of Critical Paths" CVPR 2021.

##### Licensed under the Apache License, Version 2.0 (the "License");
"""

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""## Install"""

# !pip install --quiet git+https://github.com/greentfrapp/lucent.git

"""Similar to [Lucid](https://github.com/tensorflow/lucid/), this tutorial will be focused on the same InceptionV1 model, also known as GoogLeNet.

Check out the original GoogLeNet paper [here](https://research.google.com/pubs/archive/43022.pdf).

[Distill](https://distill.pub) also has a fascinating [article](https://distill.pub/2017/feature-visualization/) on this topic that includes visualizations of all the InceptionV1 neurons.

"""

import torch
from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import vgg16, resnet50
import sys
sys.path.append('./')
import Pruner2 as Pruner
from src import Plot_tools
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

def load_img_and_model(dir):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = vgg16(pretrained=True)
  _ = model.to(device).eval()

  transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  imagenet = datasets.ImageFolder(dir, transform=transform)
  dataloader = torch.utils.data.DataLoader(imagenet, batch_size=1, shuffle=True)
  dataiter = iter(dataloader)
  data, _ = dataiter.next()
  data = data.to(device)

  class_id = model(data)
  class_id = class_id.data.max(1)[1].item()
  return model, data, device, class_id

"""## Visualize!

Below is the code to reproduce the results for the first experiment: visualizing the image which maximizes the true logit when the network is restricted only to paths selected by different methods.
"""

def apply_path(method, dir, pruner):
  # _, data, device, class_id = load_img_and_model(dir)
  
  pruner.reset()
  # pruner = Pruner.Pruner(model, data, device)
  if method == 'NeuronMCT':
    pruner.prune_neuron_mct(98, debug=False) #12855 #48083
  elif method == 'NeuronIntGrad':
    pruner.prune_integrad(98, debug=False) #12886 #48084
  elif method == 'DGR(init=1)':
    pruner.prune_dgr(98, debug=False) #64238 #51010
  elif method == 'DGR(init=r)':
    pruner.prune_dgr(98, r=True, debug=False) #64238 #51010
  elif method == 'GreedyPruning':
    pruner.prune_greedy(1, 98, debug=False) #12886 #27307
  elif method == 'RandomPruning':
    pruner.prune_random(98, debug=False)
  elif method == 'Active Subnet':
    x = int(pruner.base_sparsity()*100)+1
    pruner.prune_neuron_mct(x, debug=False)
  elif method == 'Original Net':
    pass

def vis(method, dir, pruner):
  # A visualization with gradient descent in pixel space
  # Notice the high frequency components similar to adversarial images
  if not method == 'Original Net':
    m = pruner.model
  else:
    m, _, _, _ = load_img_and_model(dir)
  class_id = pruner.model(pruner.input)
  class_id = class_id.data.max(1)[1].item()
  apply_path(method, dir, pruner)
  obj = objectives.channel("classifier", class_id)
  param_f = lambda: param.image(128, fft=False, decorrelate=False)
  # We set transforms=[] to denote no transforms
  pic = render.render_vis(m, obj, param_f, transforms=[])
  return pic

def visualize_maximizing_img(methods, dir):
  model, data, device, class_id = load_img_and_model(dir)
  pruner = Pruner.Pruner(model, data, device)
  image = Plot_tools.reverse_preprocess_imagenet_image(data)
  n = len(methods)
  pics = np.empty([n, 128, 128, 3])
  for i, m in enumerate(methods):
    pics[i] = np.asarray(vis(m, dir, pruner)).reshape(128, 128, 3)
  fig = plt.figure() #figsize=(n+1, 1))
  ax = fig.add_subplot(1, n+1, 1)
  ax.set_title("Original Image", fontsize=6)
  plt.imshow(image)
  plt.axis('off')
  for i, p in enumerate(pics):
    ax = fig.add_subplot(1, n+1, i+1+1)
    ax.set_title(methods[i], fontsize=6)
    plt.imshow(p)
    plt.axis('off')
  plt.tight_layout()
  plt.subplots_adjust(wspace=0.05, hspace=0.01)
  plt.savefig('./maximizing_img.png')
  
methods = ['Original Net', 'Active Subnet', 'NeuronIntGrad', 'NeuronMCT', 'DGR(init=1)', 'DGR(init=r)', 'GreedyPruning']
visualize_maximizing_img(methods, './path_decoding/maximizing_img/')

"""Below is the code to reproduce the results for the second experiment: visualizing the top neuron in the last layer before the linear classifier selected by different methods, which score neurons' importance. """

def visualize_top_neuron(dir, idxs, methods):
  model, data, device, _ = load_img_and_model(dir)
  image = Plot_tools.reverse_preprocess_imagenet_image(data)
  n = len(idxs)
  pics = np.empty([n, 128, 128, 3])
  for i in range(n):
    obj = objectives.channel("features",  idxs[i]) 
    pic = render.render_vis(model, obj, show_inline=False)
    pics[i] = np.asarray(pic).reshape(128, 128, 3)
  fig = plt.figure() #figsize=(n+1, 1))
  ax = fig.add_subplot(1, n+1, 1)
  ax.set_title("Original Image", fontsize=6)
  plt.imshow(image)
  plt.axis('off')
  for i, p in enumerate(pics):
    ax = fig.add_subplot(1, n+1, i+1+1)
    ax.set_title(methods[i], fontsize=6)
    plt.imshow(p)
    plt.axis('off')
  plt.tight_layout()
  plt.subplots_adjust(wspace=0.05, hspace=0.01)
  plt.savefig('./top_neuron_vis.png')


# NeuronIntGrad = 48084 -> 245
# NeuronMCT = 48083 -> 245
# DGR(init=1) = 51010 -> 260 
# DGR(init=r) = 74297 -> 379
# Greedy = 27307 -> 139

visualize_top_neuron('./path_decoding/top_neuron_vis/', [245, 245, 260, 379, 139], ['NeuronIntGrad', 'NeuronMCT', 'DGR(init=1)', 'DGR(init=r)', 'GreedyPruning'])
