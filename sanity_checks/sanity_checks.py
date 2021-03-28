import os
import random
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from skimage.feature import hog
from skimage.measure import _structural_similarity as ssim
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from tqdm import tqdm
import sys
sys.path.append('./')
from TorchRay.torchray.attribution.grad_cam import grad_cam
from src import Pruner
from src.attribution_methods import grad_times_image, guided_backprop, vanilla_saliency, integrated_gradients

device = 'cuda:0'
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
torch.random.manual_seed(12)
random.seed(12)
torch.backends.cudnn.deterministic = True
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset = datasets.ImageFolder('/home/soroosh/data', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()


class SanityCheck:
    def __init__(self, model, number_of_samples, dataloader, gradcam_layer, sparsity_level, make_single_channel=True):
        self.make_single_channel = make_single_channel
        self.model = model
        self.number_of_samples = number_of_samples
        self.dataiter = iter(dataloader)
        self.image = None
        self.label = None
        self.load_random_weights = False
        self.results = {}
        self.avg_spearman = {}
        self.avg_spearman_abs = {}
        self.spearman_rank_correlations = {}
        self.spearman_rank_correlations_abs = {}
        self.HOGs = {}
        self.pearson_HOGs = {}
        self.avg_pearson_HOGs = {}
        self.SSIMs = {}
        self.avg_SSIMs = {}
        self.gradcam_layer = gradcam_layer
        self.sparsity_level = sparsity_level
        self.comparison_path = None
        self.spearman_path = None
        self.current_sample_id = 0

    def weight_randomization(self):
        all_mods = []
        for mod in self.model.modules():
            all_mods.append(mod)
        self.get_all_saliency_maps()
        for i in range(len(all_mods)-1, -1, -1):
            module = all_mods[i]
            change = False
            if hasattr(module, 'weight'):
                change = True
                if not self.load_random_weights:
                    torch.nn.init.normal_(module.weight, 0., 0.1)
            if hasattr(module, 'bias'):
                if self.load_random_weights and (all_mods[i-1].__class__.__name__ == "ResNet" or all_mods[i-1].__class__.__name__ == "Bottleneck" or module.__class__.__name__ == "Linear"):
                    self.model.load_state_dict(torch.load("./sanity_checks/random_models/model"+str(i)+".pth"))
                    self.model.eval()
                else:
                    if module.bias is not None:
                        torch.nn.init.normal_(module.bias, 0., 0.1)
            if change and (all_mods[i-1].__class__.__name__ == "ResNet" or all_mods[i-1].__class__.__name__ == "Bottleneck" or module.__class__.__name__ == "Linear"):
                if not self.load_random_weights:
                    torch.save(self.model.state_dict(), "./sanity_checks/random_models/model"+str(i)+".pth")
                self.get_all_saliency_maps()

        self.calc_spearman()
        self.calc_HOG()
        self.calc_SSIM()
        # self.save_image_and_saliency()
        self.archive()

    def get_all_saliency_maps(self):
        all_keys = []
        for k in self.results.keys():
            all_keys.append(k)
        if len(all_keys) <= 0:
            self.results['Vanilla Gradient'] = []
            self.results['NeuronIntGrad'] = []
            self.results['InputMCT'] = []
            self.results['Guided Backprop'] = []
            self.results['InputIntGrad'] = []
            self.results['Grad Cam'] = []
            self.results['NeuronMCT'] = []

        self.model.eval()
        class_id = self.model(self.image)
        class_id = class_id.data.max(1)[1].item()
        for k in self.results.keys():
            if k == "Vanilla Gradient":
                vanilla_sal = vanilla_saliency.VanillaSaliency(self.model, device)
                saliency = vanilla_sal.generate_saliency(self.image, class_id, self.make_single_channel)
            if k == "InputMCT":
                # VanillaGrad Should be active for running Grad*Inp (saliency argumenr is from VanillaGrad's result)
                saliency = grad_times_image.generate_grad_times_image_saliency(self.model, self.image, self.label, device, self.make_single_channel, saliency)
            elif k == "Guided Backprop":
                GB = guided_backprop.GuidedBackprop(self.model, device)
                saliency = GB.generate_gradients(self.image, class_id, self.make_single_channel)
            elif k == "InputIntGrad":
                integ_grad = integrated_gradients.IntegratedGradients(self.model, device)
                saliency = integ_grad.generate_integrated_gradients(self.image, class_id, 50, self.make_single_channel)
            elif k == "Grad Cam":
                saliency = grad_cam(self.model, self.image, class_id, saliency_layer=self.gradcam_layer)
                saliency = F.interpolate(saliency, 224, mode="bilinear")
                saliency = saliency.detach().cpu().numpy()
            elif k == "NeuronMCT":
                pruner = Pruner.Pruner(self.model, self.image, device)
                pruner.prune_neuron_mct(self.sparsity_level)
                saliency = pruner.generate_saliency(make_single_channel=self.make_single_channel)
                pruner.remove_handles()
            elif k == "NeuronIntGrad":
                pruner = Pruner.Pruner(self.model, self.image, device)
                pruner.prune_integrad(self.sparsity_level)
                saliency = pruner.generate_saliency(make_single_channel=self.make_single_channel)
                pruner.remove_handles()

            if k == "Grad Cam" or self.make_single_channel:
                saliency = torch.from_numpy(np.asarray(saliency)).view([1, 224, 224])
            else:
                saliency = torch.from_numpy(np.asarray(saliency)).view([3, 224, 224])
            saliency /= np.max(np.asarray(abs(saliency)).flatten())
            self.results[k].append(saliency)

    def save_image_and_saliency(self):
        keys = []
        for k in self.results.keys():
            keys.append(k)

        number_of_methods = len(self.results)
        number_of_layers = len(self.results['Vanilla Gradient'])

        image_np = self.image.detach().cpu().numpy()
        image_np[0][0] = image_np[0][0] * 0.229 + 0.485
        image_np[0][1] = image_np[0][1] * 0.224 + 0.456
        image_np[0][2] = image_np[0][2] * 0.225 + 0.406

        fig = plt.figure()

        for i in range(number_of_methods):
            method = keys[i]
            for j in range(number_of_layers):
                fig.add_subplot(number_of_methods, number_of_layers+1, i*(number_of_layers+1)+j+2)
                image_pos = deepcopy(np.asarray(self.results[method][j].cpu().squeeze(0)))
                image_pos[image_pos < 0.] = 0.
                image_neg = deepcopy(np.asarray(self.results[method][j].cpu().squeeze(0)))
                image_neg[image_neg > 0.] = 0.
                image_neg = abs(image_neg)
                if image_pos.shape[0] == 3:
                    plt.imshow(image_pos.transpose((1, 2, 0)), cmap='Reds')# , vmin=min_pixel, vmax=max_pixel)
                    plt.imshow(image_neg.transpose((1, 2, 0)), cmap='Blues', alpha=0.5)# , vmin=min_pixel, vmax=max_pixel, alpha=0.5)
                else:
                    plt.imshow(image_pos, cmap='Reds')# , vmin=min_pixel, vmax=max_pixel)
                    plt.imshow(image_neg, cmap='Blues', alpha=0.5)# , vmin=min_pixel, vmax=max_pixel, alpha=0.5)
                plt.axis('off')
            ax = fig.add_subplot(number_of_methods, number_of_layers+1, i*(number_of_layers+1)+1)
            plt.imshow(np.asarray(image_np.squeeze(0).transpose(1, 2, 0)))
            plt.text(0., 0.5, method, ha='right', va='bottom', fontsize=2, transform=ax.transAxes)
            plt.axis('off')

        if self.comparison_path is None:
            now = str(datetime.now()).replace(':', '-')
            if not os.path.isdir("./sanity_checks/saliency_comparisons/"+now+"/"):
                try:
                    os.makedirs("./sanity_checks/saliency_comparisons/"+now+"/")
                except OSError:
                    print("Directory creation FAILED!", OSError)
            self.comparison_path = r"./sanity_checks/saliency_comparisons/" + now + "/"
        plt.savefig(self.comparison_path + str(self.current_sample_id) + ".svg", format='svg', dpi=1200)
        plt.close()

    def calc_spearman(self):
        for k in self.results.keys():
            self.spearman_rank_correlations[k] = []
            self.spearman_rank_correlations_abs[k] = []
            for j in range(len(self.results[k])):
                res = stats.spearmanr(self.results[k][0].cpu().view(-1), self.results[k][j].cpu().view(-1))
                res_abs = stats.spearmanr(abs(self.results[k][0].cpu().view(-1)), abs(self.results[k][j].cpu().view(-1)))
                self.spearman_rank_correlations[k].append(res[0])
                self.spearman_rank_correlations_abs[k].append(res_abs[0])

    def calc_HOG(self):
        for k in self.results.keys():
            self.HOGs[k] = []
            for j in range(len(self.results[k])):
                self.HOGs[k].append(hog(np.asarray(self.results[k][j].cpu()).transpose(1, 2, 0), pixels_per_cell=(16, 16), multichannel=True))
        for k in self.results.keys():
            self.pearson_HOGs[k] = []
            for j in range(len(self.HOGs[k])):
                res = stats.pearsonr(self.HOGs[k][0], self.HOGs[k][j])
                self.pearson_HOGs[k].append(res[0])

    def calc_SSIM(self):
        for k in self.results.keys():
            self.SSIMs[k] = []
            for j in range(len(self.results[k])):
                res = ssim.compare_ssim(np.asarray(self.results[k][0].cpu()).transpose(1, 2, 0), np.asarray(self.results[k][j].cpu()).transpose(1, 2, 0), win_size=5, multichannel=True)
                self.SSIMs[k].append(res)

    def archive(self):
        spearman = self.spearman_rank_correlations
        spearman_abs = self.spearman_rank_correlations_abs
        pearson_HOG = self.pearson_HOGs
        ssim = self.SSIMs
        for k in spearman.keys():
            if k not in self.avg_spearman.keys():
                self.avg_spearman[k] = np.asarray(spearman[k])
                self.avg_spearman_abs[k] = np.asarray(spearman_abs[k])
                self.avg_pearson_HOGs[k] = np.asarray(pearson_HOG[k])
                self.avg_SSIMs[k] = np.asarray(ssim[k])
            else:
                self.avg_spearman[k] = np.add(self.avg_spearman[k], np.asarray(spearman[k]))
                self.avg_spearman_abs[k] = np.add(self.avg_spearman_abs[k], np.asarray(spearman_abs[k]))
                self.avg_pearson_HOGs[k] = np.add(self.avg_pearson_HOGs[k], np.asarray(pearson_HOG[k]))
                self.avg_SSIMs[k] = np.add(self.avg_SSIMs[k], np.asarray(ssim[k]))
        self.spearman_rank_correlations = {}
        self.spearman_rank_correlations_abs = {}
        self.HOGs = {}
        self.pearson_HOGs = {}
        self.SSIMs = {}
        self.results = {}
        self.current_sample_id += 1

    def plot_results(self, num_plot):
        to_be_plot = [self.avg_spearman, self.avg_spearman_abs, self.avg_SSIMs, self.avg_pearson_HOGs]
        name_of_plot = ["SPR", "ABS SPR", "HOG", "SSIM"]
        if self.spearman_path is None:
            now = str(datetime.now()).replace(':', '-')
            if not os.path.isdir("./sanity_checks/plots/" + now):
                try:
                    os.makedirs("./sanity_checks/plots/" + now + "/")
                except OSError:
                    print("Directory creation FAILED!", OSError)
            self.spearman_path = r"./sanity_checks/plots/" + now + "/"
        saved_k = None
        for i in range(len(to_be_plot)):
            plt.figure()
            for k in to_be_plot[i].keys():
                saved_k = k
                if len(to_be_plot[i][k].shape) == 1:
                    tmp = np.divide(to_be_plot[i][k], num_plot)
                    np.save(self.spearman_path+k+str(i)+'-numberOfSamples'+str(num_plot)+'.npy', np.asarray(tmp))
                    plt.plot(tmp, label=str(k))
                else:
                    tmp = np.divide(to_be_plot[i][k][:, 0], num_plot)
                    np.save(self.spearman_path+k+str(i)+'-numberOfSamples'+str(num_plot)+'.npy', np.asarray(tmp))
                    plt.plot(tmp, label=str(k))
                plt.legend()
                plt.axis([0, len(to_be_plot[i][k])+1, -1, 1])
                plt.ylabel(name_of_plot[i])
            plt.savefig(self.spearman_path + "plot" + str(name_of_plot[i]) + "_plot_with_number_" + str(num_plot) + ".svg", format='svg', dpi=1200)

    def reset_model(self, load_random_weights):
        model = models.resnet50(pretrained=True)
        model = model.to(device)
        model.eval()
        self.model = model
        self.load_random_weights = load_random_weights

    def run(self, load_random_weights):
        plot_interval = 10
        for i in tqdm(range(self.number_of_samples)):
            image, label = self.dataiter.next()
            self.image = image.to(device)
            self.label = label.to(device)
            self.reset_model(load_random_weights)
            self.weight_randomization()
            load_random_weights = True
            if (i+1) % plot_interval == 0:
                self.plot_results(i+1)
        self.plot_results(self.number_of_samples)


gradcam_resnet_layer = 'layer4'
s = SanityCheck(model, 1000, dataloader, gradcam_resnet_layer, 70)
s.run(False)
