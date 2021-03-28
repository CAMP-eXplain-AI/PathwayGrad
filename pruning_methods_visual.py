import matplotlib as mpl
import numpy as np
import torch
from torchvision import datasets, models, transforms
from tqdm import tqdm
from src import Pruner, Plot_tools
from src.attribution_methods import vanilla_saliency
import matplotlib.pyplot as plt

# ### Setup Imagenet
# 
# ImageNet as of Oct2019 can no longer be downloaded using pytorch.  
# https://github.com/pytorch/vision/issues/1453  
# To download ImageNet, see http://image-net.org/.  

imagenet_dir = '/home/ashkan/data/ILSVRC2012/'
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
imagenet = datasets.ImageNet(imagenet_dir, download=False, split='val', transform=transform)
classes = imagenet.classes
mpl.rcParams['figure.dpi']= 400


# ### Method to get attributions

def get_attribution(attribution_name, data, model, model_sparsity_threshold):
    make_single_channel = True
    class_id = model(data)
    class_id = class_id.data.max(1)[1].item()
    # print(class_id)
    model.eval()
    if attribution_name == "Gradients":
        vanilla_sal = vanilla_saliency.VanillaSaliency(model, device)
        saliency = vanilla_sal.generate_saliency(data, class_id, make_single_channel)
    
    elif attribution_name == "NeuronMCT":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_neuron_mct(model_sparsity_threshold, debug=False)
        saliency = pruner.generate_saliency(make_single_channel=make_single_channel)
        pruner.remove_handles()

    elif attribution_name == "NeuronIntGrad":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_integrad(model_sparsity_threshold, debug=False)
        saliency = pruner.generate_saliency(make_single_channel=make_single_channel)
        pruner.remove_handles()

    if make_single_channel:
        saliency = torch.from_numpy(np.asarray(saliency)).view([1, 224, 224])
    else:
        saliency = torch.from_numpy(np.asarray(saliency)).view([3, 224, 224])
    saliency /= np.max(np.asarray(abs(saliency)).flatten())
    return saliency


def visualize(model, dataloader, images, classes, sparsity_levels):
    global id
    model = model.to(device)
    model.eval()
    name_methods = ["NeuronIntGrad", "NeuronMCT"]
    num_methods = len(name_methods)
    dataiter = iter(dataloader)
    i = 0
    fig = plt.figure(figsize=(7, num_samples*2))
    acts = []
    for chosen in tqdm(range(num_samples)):
        data, _ = dataiter.next()
        data = data.to(device)
        output = model(data.clone())
        output = torch.nn.functional.softmax(output.detach(), dim=1)
        predicted_logit = output.data.max(1)[1].item()
        predicted_prob = output.data.max(1)[0].item()
        image = Plot_tools.reverse_preprocess_imagenet_image(data.clone())
        ax = fig.add_subplot(num_samples*num_methods, len(sparsity_levels)+1, 2*chosen*(len(sparsity_levels)+1) + 1)
        if chosen == 0:
            ax.set_title("Original Image", fontsize=6)
        ax.text(-0.13, 0.5, classes[predicted_logit][0]+"\n"+str("%.2f" % round(predicted_prob*100, 2)+"%"), fontsize=6, rotation=90, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.axis('off')
        plt.imshow(image)
        for j in range(num_methods):
            for k in range(len(sparsity_levels)):
              ax = fig.add_subplot(num_samples*num_methods, len(sparsity_levels)+1, num_methods*chosen*(len(sparsity_levels)+1) + j*(len(sparsity_levels)+1) + 1 + k + 1)
              if sparsity_levels[k] == 0:
                  attribution = get_attribution('Gradients', data.clone(), model, sparsity_levels[k])
              else:
                  attribution = get_attribution(name_methods[j], data.clone(), model, sparsity_levels[k])
              attribution = np.asarray(attribution.squeeze(0))
              if chosen == 0 and j == 0:
                if sparsity_levels[k] != 0:
                  ax.set_title("Sparsity={}".format(sparsity_levels[k]), fontsize=6)
                else:
                  ax.set_title("Original Network", fontsize=6)
              plt.imshow(abs(attribution), cmap='jet', vmin=0, vmax=1)
              if k == len(sparsity_levels)-1:
                  ax.text(1.05, 0.5, name_methods[j], fontsize=5, rotation=-90, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontweight='bold')
              plt.axis('off')
        i += 1
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.01)
    plt.savefig('./different_sparsity_'+str(id)+'.png', dpi=300)
    id += 1


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

num_samples = 1
id = 0
indices = [16305] #butterfly
dataset = torch.utils.data.Subset(imagenet, indices)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)


# # Evaluate on VGG16

dataset = torch.utils.data.Subset(imagenet, indices)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

model = models.vgg16(pretrained=True)
model_sparsity_threshold = 90 # Threshold computed for 15% output change for Pruner
visualize(model, dataloader, indices, classes, [0, 70, 80, 85, 90, 95, 99])

