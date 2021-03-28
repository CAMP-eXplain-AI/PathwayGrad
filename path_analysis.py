import random
import matplotlib as mpl
import numpy as np
import torch
from torchvision import datasets, models, transforms
from tqdm import tqdm
from src import Pruner
import os 

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
mpl.rcParams['figure.dpi'] = 1200


def dead_path_analysis(dead_path, path):
    return (np.dot(path, dead_path)/sum(path))


def iou_path_analysis(path1, path2):
    return (np.dot(path1, path2)/(sum(path1+path2)-np.dot(path1, path2)))


def get_path(attribution_name, data, model, model_sparsity_threshold):
    class_id = model(data)
    model.eval()

    if attribution_name == "NC":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_neuron_mct(model_sparsity_threshold, debug=False)
        path = pruner.pruned_activations_mask
        pruner.remove_handles()

    elif attribution_name == "IntGrad":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_integrad(model_sparsity_threshold, debug=False)
        path = pruner.pruned_activations_mask
        pruner.remove_handles()

    elif attribution_name == "GreedyNC":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_greedy(1, 99)
        path = pruner.pruned_activations_mask
        pruner.remove_handles()

    elif attribution_name == "Random":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_random(model_sparsity_threshold, debug=False)
        path = pruner.pruned_activations_mask
        pruner.remove_handles()

    elif attribution_name == "CDRP":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_dgr(model_sparsity_threshold, debug=False)
        path = pruner.pruned_activations_mask
        pruner.remove_handles()

    elif attribution_name == "CDRP-R":
        pruner = Pruner.Pruner(model, data, device)
        pruner.prune_dgr(model_sparsity_threshold, debug=False, r=True)
        path = pruner.pruned_activations_mask
        pruner.remove_handles()

    elif attribution_name == "DeadsPath":
        pruner = Pruner.Pruner(model, data, device)
        pruner.dead_neurons_path()
        path = pruner.pruned_activations_mask
        pruner.remove_handles()

    layer_paths = []
    for i in range(len(path)):
      layer_paths.append(path[i].clone().cpu().detach().numpy().reshape(-1))

    path = np.hstack((layer_paths[i] for i in range(len(layer_paths))))
    return path, layer_paths


def analyze_paths(model, dataloader, images, classes, model_sparsity_threshold):
    global id
    base_dir = './paths_logs_'+str(model_sparsity_threshold)+'/'
    print(base_dir)
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    model = model.to(device)
    model.eval()
    name_methods = ["CDRP", "NC", "IntGrad", "GreedyNC", "Random", "CDRP-R"]
    num_methods = len(name_methods)
    dataiter = iter(dataloader)
    i = 0
    iou = [[0 for j in range(num_methods)] for k in range(num_methods)]
    dead_portions = [0 for j in range(num_methods)]
    samples_ious = np.zeros((num_samples, num_methods, num_methods))
    samples_deads = np.zeros((num_samples, num_methods))
    layer_based_iou = None
    layer_based_dead = None
    samples_layer_ious = None
    samples_layer_deads = None
    n_layers = None
    for chosen in tqdm(range(num_samples)):
        data, _ = dataiter.next()
        data = data.to(device)
        output = model(data.clone())
        output = torch.nn.functional.softmax(output.detach(), dim=1)
        predicted_logit = output.data.max(1)[1].item()
        predicted_prob = output.data.max(1)[0].item()
        paths = [None for j in range(num_methods)]
        layer_paths = [None for j in range(num_methods)]
        new_iou = [[0 for j in range(num_methods)] for k in range(num_methods)]
        new_deads = [0 for j in range(num_methods)]
        for j in range(num_methods):
            paths[j], layer_paths[j] = get_path(name_methods[j], data.clone(), model, model_sparsity_threshold)
            if n_layers is None:
                n_layers = len(layer_paths[j])
                layer_based_iou = [[[0 for l in range(n_layers)] for m in range(num_methods)] for k in range(num_methods)]
                layer_based_dead = [[0 for l in range(n_layers)] for k in range(num_methods)]
                samples_layer_ious = np.zeros((num_samples, num_methods, num_methods, n_layers))
                samples_layer_deads = np.zeros((num_samples, num_methods, n_layers))
        new_layer_ious = [[[0 for l in range(n_layers)] for j in range(num_methods)] for k in range(num_methods)]
        new_layer_dead = [[0 for l in range(n_layers)] for k in range(num_methods)]
        for j in range(num_methods):
            for k in range(j, num_methods):
                new_iou[j][k] = iou_path_analysis(paths[j], paths[k])
                iou[j][k] += new_iou[j][k]
                for l in range(n_layers):
                    new_layer_ious[j][k][l] = iou_path_analysis(layer_paths[j][l], layer_paths[k][l])
                    layer_based_iou[j][k][l] += new_layer_ious[j][k][l]
                #print(name_methods[j], "vs.", name_methods[k], iou_path_analysis(paths[j], paths[k]))
        dead_path, l_dead_path = get_path("DeadsPath", data.clone(), model, 0)
        #print("Dead Neuron Analysis")
        for j in range(num_methods):
            new_deads[j] = dead_path_analysis(dead_path, paths[j])
            dead_portions[j] += new_deads[j]
            for l in range(n_layers):
                new_layer_dead[j][l] = dead_path_analysis(l_dead_path[l], layer_paths[j][l])
                layer_based_dead[j][l] += new_layer_dead[j][l]

            #print("Dead Portion of", name_methods[j], dead_path_analysis(dead_path, paths[j]))
        samples_ious[chosen] = np.asarray(new_iou)
        samples_deads[chosen] = np.asarray(new_deads)
        samples_layer_ious[chosen] = np.asarray(new_layer_ious)
        samples_layer_deads[chosen] = np.asarray(new_layer_dead)

        if chosen % 50 == 0:
            saved_iou = np.asarray(iou)/(chosen+1)
            saved_dead = np.asarray(dead_portions)/(chosen+1)
            saved_layer_ious = np.asarray(layer_based_iou)/(chosen+1)
            saved_layer_deads = np.asarray(layer_based_dead)/(chosen+1)
            print(saved_iou)
            print(saved_dead)
#            print(samples_ious)
#            print(samples_deads)
            print(saved_layer_ious)
            print(saved_layer_deads)
            np.save(base_dir + '/jaccards'+str(chosen)+'.npy', saved_iou)
            np.save(base_dir + '/deads'+str(chosen)+'.npy', saved_dead)
            np.save(base_dir + '/layerwise_jaccards'+str(chosen)+'.npy', saved_layer_ious)
            np.save(base_dir + '/layerwise_dead'+str(chosen)+'.npy', saved_layer_deads)
            np.save(base_dir + '/samples_jaccards'+str(chosen)+'.npy', samples_ious)
            np.save(base_dir + '/samples_deads'+str(chosen)+'.npy', samples_deads)
            np.save(base_dir + '/samples_layerwise_jaccards'+str(chosen)+'.npy', samples_layer_ious)
            np.save(base_dir + '/samples_layerwise_deads'+str(chosen)+'.npy', samples_layer_deads)

        i += 1

    iou = np.asarray(iou)/num_samples
    dead_portions = np.asarray(dead_portions)/num_samples
    layer_based_iou = np.asarray(layer_based_iou)/num_samples
    layer_based_dead = np.asarray(layer_based_dead)/num_samples
#    print(layer_based_iou)
    np.save(base_dir + '/jaccards'+str('final')+'.npy', iou)
    np.save(base_dir + '/deads'+str('final')+'.npy', dead_portions)
    np.save(base_dir + '/layerwise_jaccards'+str('final')+'.npy', layer_based_iou)
    np.save(base_dir + '/layerwise_deads'+str('final')+'.npy', layer_based_dead)


    np.save(base_dir + '/samples_jaccards'+str('final')+'.npy', samples_ious)
    np.save(base_dir + '/samples_deads'+str('final')+'.npy', samples_deads)
    np.save(base_dir + '/samples_layerwise_jaccards'+str('final')+'.npy', samples_layer_ious)
    np.save(base_dir + '/samples_layerwise_deads'+str('final')+'.npy', samples_layer_deads)


num_samples = 1000

indices = random.sample(range(0, len(imagenet)), num_samples)
dataset = torch.utils.data.Subset(imagenet, indices)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

model = models.vgg16(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_sparsity_threshold = 80 # 90, 99
analyze_paths(model, dataloader, indices, classes, model_sparsity_threshold)

