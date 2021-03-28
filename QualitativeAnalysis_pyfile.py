import random
from torch.nn import functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, models, transforms
from tqdm import tqdm
from torchray.attribution.grad_cam import grad_cam
from src import Pruner, Plot_tools
from src.attribution_methods import vanilla_saliency, integrated_gradients, grad_times_image, guided_backprop

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
mpl.rcParams['figure.dpi']= 1200


# ### Method to get attributions

def get_attribution(attribution_name, data, model, gradcam_layer, model_sparsity_threshold):
    make_single_channel = True
    class_id = model(data)
    class_id = class_id.data.max(1)[1].item()
    # print(class_id)
    model.eval()
    if attribution_name == "Gradients":
        vanilla_sal = vanilla_saliency.VanillaSaliency(model, device)
        saliency = vanilla_sal.generate_saliency(data, class_id, make_single_channel)
    elif attribution_name == "InputMCT":
        saliency = grad_times_image.generate_grad_times_image_saliency(model, data, class_id, device, make_single_channel)
    elif attribution_name == "InputIntGrad":
        integ_grad = integrated_gradients.IntegratedGradients(model, device)
        saliency = integ_grad.generate_integrated_gradients(data, class_id, 50, make_single_channel)
    elif attribution_name == "GBP":
        GB = guided_backprop.GuidedBackprop(model, device)
        saliency = GB.generate_gradients(data, class_id, make_single_channel)

    elif attribution_name == "GradCAM":
        saliency = grad_cam(model, data, class_id, saliency_layer=gradcam_layer)
        saliency = F.interpolate(saliency, 224, mode="bilinear")
        saliency = saliency.detach().cpu().numpy()

    elif attribution_name == "NeuronMCT":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_neuron_mct(model_sparsity_threshold, debug=False)
        saliency = pruner.generate_saliency(make_single_channel=make_single_channel)
        pruner.remove_handles()

    elif attribution_name == "PrunedRandom":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_random(model_sparsity_threshold, debug=False)
        saliency = pruner.generate_saliency(make_single_channel=make_single_channel)
        pruner.remove_handles()

    elif attribution_name == "NeuronIntGrad":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_integrad(model_sparsity_threshold, debug=False)
        saliency = pruner.generate_saliency(make_single_channel=make_single_channel)
        pruner.remove_handles()

    elif attribution_name == "PruneGreedy":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_greedy(1, model_sparsity_threshold, debug=False)
        saliency = pruner.generate_saliency(make_single_channel=make_single_channel)
        pruner.remove_handles()

    elif attribution_name == "CDRP":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_dgr(model_sparsity_threshold, debug=False)
        saliency = pruner.generate_saliency(make_single_channel=make_single_channel)
        pruner.remove_handles()

    elif attribution_name == "PrunePGD":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_neuron_mct(model_sparsity_threshold, debug=False)
        saliency = pruner.generate_saliency_pgd_l2(epsilon=500, alpha=25, num_iter=50, make_single_channel=make_single_channel, debug=False)
        pruner.remove_handles()

    if make_single_channel:
        saliency = torch.from_numpy(np.asarray(saliency)).view([1, 224, 224])
    else:
        saliency = torch.from_numpy(np.asarray(saliency)).view([3, 224, 224])
    saliency /= np.max(np.asarray(abs(saliency)).flatten())
    return saliency


def visualize(model, dataloader, images, classes, gradcam_layer, model_sparsity_threshold):
    global id
    model = model.to(device)
    model.eval()
    name_methods = ["Gradients", "GBP", "GradCAM", "InputMCT", "InputIntGrad", "NeuronMCT", "NeuronIntGrad"] 
    num_methods = len(name_methods)
    dataiter = iter(dataloader)
    i = 0
    fig = plt.figure(figsize=(7, num_samples)) #figsize=(7, 10*2))
    acts = []
    for chosen in tqdm(range(num_samples)):
        data, _ = dataiter.next()
        data = data.to(device)
        output = model(data.clone())
        output = torch.nn.functional.softmax(output.detach(), dim=1)
        predicted_logit = output.data.max(1)[1].item()
        predicted_prob = output.data.max(1)[0].item()
        image = Plot_tools.reverse_preprocess_imagenet_image(data.clone())
        ax = fig.add_subplot(num_samples, num_methods+1, chosen*(num_methods+1) + 1)
        if chosen == 0:
            ax.set_title("Original Image", fontsize=7)
        ax.text(-0.13, 0.5, classes[predicted_logit][0]+"\n"+str("%.2f" % round(predicted_prob*100, 2)+"%"), fontsize=7, rotation=90, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.axis('off')
        plt.imshow(image)
        for j in range(num_methods):
            ax = fig.add_subplot(num_samples, num_methods+1, chosen*(num_methods+1) + 1 + j + 1)
            attribution = get_attribution(name_methods[j], data.clone(), model, gradcam_layer,  model_sparsity_threshold)
            attribution = np.asarray(attribution.squeeze(0))
            if chosen == 0:
                if name_methods[j] == "NeuronMCT" or name_methods[j] == "NeuronIntGrad":
                    ax.set_title(name_methods[j], fontsize=7, fontweight='bold')
                else:
                    ax.set_title(name_methods[j], fontsize=7)
            plt.imshow(abs(attribution), cmap='jet', vmin=0, vmax=1)
            plt.axis('off')
        i += 1
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.01)
    plt.savefig('appendix_attribution_'+str(indices)+'.png', dpi=200) #'./fig'+str(id)+'.png', dpi=200)
    id += 1


# # Evaluate on Resnet50

num_samples = 9
id = 0
indices = [5054, 36203, 43518, 20813, 7054, 15574, 15598, 41442, 9678] #RESNET APPENDIX
dataset = torch.utils.data.Subset(imagenet, indices)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

model = models.resnet50(pretrained=True)
device = 'cuda:0'
resnet_gradcam_layer = 'layer4'
model_sparsity_threshold = 75
visualize(model, dataloader, indices, classes, resnet_gradcam_layer, model_sparsity_threshold)


# # Evaluate on VGG16

num_samples = 9 #1
# indices = [40639] # airplane
indices = [39834, 40639, 3158, 8105, 2094, 23684, 12659, 23147, 4581]  #VGG16 APPENDIX
dataset = torch.utils.data.Subset(imagenet, indices)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

model = models.vgg16(pretrained=True)
vgg_gradcam_layer = 'features'
model_sparsity_threshold = 90
visualize(model, dataloader, indices, classes, vgg_gradcam_layer, model_sparsity_threshold)

