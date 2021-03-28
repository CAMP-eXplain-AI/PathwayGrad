import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


def load_mean_paths(load_dir):
    m = np.load(load_dir)
    for i in range(m.shape[0]-1):
        for j in range(i+1, m.shape[1]):
            m[j][i] = m[i][j]
    m_c = m.copy()
    m[0], m[1], m[2], m[3], m[4], m[5] = m_c[1], m_c[2], m_c[0], m_c[5], m_c[4], m_c[3]
    m_c = m.copy()
    m[:, 0], m[:, 1], m[:, 2], m[:, 3], m[:, 4], m[:, 5] = m_c[:, 1], m_c[:, 2], m_c[:, 0], m_c[:, 5], m_c[:, 4], m_c[:, 3]
    mask = np.zeros(m.shape)
    for i in range(m.shape[0] - 1):
        for j in range(i + 1, m.shape[1]):
            mask[j][i] = 1

    return m, mask


def make_confusion_matrix(m, s):
    categories = ['Neuron\nIntGrad', 'Neuron\nMCT', 'DGR\n(init=1)', 'DGR\n(init=r)', 'Greedy\nPruning', 'Random\nPruning']
    # sns.set(font_scale=.5)
    m, mask = m
    plt.figure(figsize=(9, 3.4))
    ax = sns.heatmap(m, annot=True, fmt='.2%', cmap='Blues', mask=mask)
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories, rotation=0)
    plt.title('Jaccard Similarity for Sparsity={}'.format(s)) #\nSparsity Level={}'.format(s))
    plt.tight_layout(0.01)
    plt.show()
    plt.savefig('./matrix_sparsity_{}'.format(s)+'.png')
    plt.close()


def make_deads_barplot(a, s):

    plt.style.use('seaborn-dark-palette')
    plt.figure(figsize=(9, 5))
    a_c = a.copy()
    a[0], a[1], a[2], a[3], a[4], a[5] = a_c[1], a_c[2], a_c[0], a_c[5], a_c[3], a_c[4]
    fake_first_plot = np.ones(a.shape)
    # Set general plot properties
    # sns.set_style("white")
    # sns.set_context({"figure.figsize": (10, 1)})

    # Plot 1 - background - "total" (top) series
    sns.barplot(x=['Neuron\nIntGrad', 'Neuron\nMCT', 'DGR\n(init=1)', 'DGR\n(init=r)', 'Greedy\nPruning', 'Random\nPruning'], y=fake_first_plot*100, color="#0C7BDC")
	#forestgreen red

    # Plot 2 - overlay - "bottom" series
    bottom_plot = sns.barplot(x=['Neuron\nIntGrad', 'Neuron\nMCT', 'DGR\n(init=1)', 'DGR\n(init=r)', 'Greedy\nPruning', 'Random\nPruning'], y=a*100, color="#FFC20A")

    topbar = plt.Rectangle((0, 0), 1, 1, fc="#0C7BDC", edgecolor='none')
    bottombar = plt.Rectangle((0, 0), 1, 1, fc='#FFC20A', edgecolor='none')
    l = plt.legend([bottombar, topbar], ['DeadNeurons', 'ActiveNeurons'], bbox_to_anchor=(0.70, 1.16), loc='upper left', ncol=1, prop={'size': 15})
    l.draw_frame(False)

    # Optional code - Make plot look nicer
    sns.despine(left=True)
    bottom_plot.set_ylabel("Percentage", fontweight='bold')
    bottom_plot.set_xlabel("Methods", fontweight='bold')
    bottom_plot.set_title('Sparsity={}'.format(s), fontweight='bold')

    # Set fonts to consistent 13pt size
    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
                 bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        item.set_fontsize(14)
    bottom_plot.title.set_fontsize(15)
    for i, v in enumerate(a):
        val = round(v, 4)*100
        bottom_plot.text(i-0.27, v*50+0.01, '{:.2f}%'.format(val), color='black', fontweight='bold', fontsize=15)
    plt.tight_layout(0.01, 0.01)
    # plt.show()
    plt.savefig('./deads_barplot_sparsity_{}'.format(s)+'.png')
    plt.close()


def roar_plot(dset):
    methods = ['Gradients', 'GBP', 'GradCAM', 'InputMCT', 'InputIntGrad', 'NeuronMCT', 'NeuronIntGrad', 'NeuronIntGrad*']
    res = []
    if dset == 'cifar10':
        initial_accuracy = [84.36]
        gradients = [79.100, 81.100, 82.810, 82.260, 75.500]
        res.append(initial_accuracy+gradients)
        gbp = [79.240, 76.830, 78.020, 78.360, 74.780]
        res.append(initial_accuracy+gbp)
        gradcam = [74.650, 69.990, 63.270, 58.030, 47.580]
        res.append(initial_accuracy+gradcam)
        input_mct = [79.220, 79.850, 82.540, 81.580, 74.470]
        res.append(initial_accuracy+input_mct)
        input_intgrad = [80.800, 81.980, 83.370, 83.230, 80.900]
        res.append(initial_accuracy+input_intgrad)
        neuron_mct = [74.430, 67.800, 65.480, 57.870, 44.390]
        res.append(initial_accuracy+neuron_mct)
        neuron_intgrad = [75.110, 67.330, 61.710, 56.900, 47.980]
        res.append(initial_accuracy+neuron_intgrad)
        neuron_intgrad_star = [75.090, 63.560, 54.600, 47.850, 37.270]
        res.append(initial_accuracy+neuron_intgrad_star)
        title = "ROAR Benchmark on CIFAR10"
    elif dset == 'birdsnap':
        initial_accuracy = [58.630]
        gradients = [33.615, 29.395, 26.823, 24.312, 28.692]
        res.append(initial_accuracy+gradients)
        gbp = [30.299, 26.160, 22.604, 23.347, 24.051]
        res.append(initial_accuracy+gbp)
        gradcam = [20.052, 7.816, 5.304, 4.240, 2.351]
        res.append(initial_accuracy+gradcam)
        input_mct = [31.565, 27.165, 23.508, 21.640, 20.836]
        res.append(initial_accuracy+input_mct)
        input_intgrad = [30.762, 26.723, 23.146, 22.343, 22.483]
        res.append(initial_accuracy+input_intgrad)
        neuron_mct= [22.885, 12.678, 8.620, 3.797, 3.074]
        res.append(initial_accuracy+neuron_mct)
        neuron_intgrad = [19.023, 9.13, 6.32, 4.01, 3.03]
        res.append(initial_accuracy+neuron_intgrad)
        neuron_intgrad_star = [16.858, 8.077, 5.210, 3.680, 2.213]
        res.append(initial_accuracy+neuron_intgrad_star)
        title = "ROAR Benchmark on Birdsnap"

    color_mapping = {
        'Gradients': 2,
        'GBP': 4,
        'GradCAM': 6,
        'InputIntGrad': 10,
        'InputMCT': 8,
        'NeuronMCT': 12,
        'NeuronIntGrad': 14,
        'NeuronIntGrad*': 18}

    tableau20 = [(31, 119, 180), (174, 199, 232),
                 (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138),
                 (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213),
                 (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210),
                 (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    aucs = {}
    plt.style.use('seaborn-dark-palette')
    fig, ax = plt.subplots(figsize=(8, 3))
    # fig, ax = plt.subplots()
    for i in range(len(methods)):
        auc = metrics.auc(np.asarray([0, 10, 30, 50, 70, 90])/100, np.asarray(res[i])/100)
        aucs[methods[i]] = auc
        if i < 5:
            ax.plot([0, 10, 30, 50, 70, 90], res[i], label=methods[i], color=tableau20[color_mapping[methods[i]]], linewidth=2, alpha=.7)
        elif methods[i] == 'NeuronMCT':
            ax.plot([0, 10, 30, 50, 70, 90], res[i], marker='*', label=r'\textbf{NeuronMCT}', color=tableau20[color_mapping[methods[i]]], linewidth=2, alpha=.7)
        elif methods[i] == 'NeuronIntGrad':
            ax.plot([0, 10, 30, 50, 70, 90], res[i], marker='*', label=r'\textbf{NeuronIntGrad}', color=tableau20[color_mapping[methods[i]]], linewidth=2, alpha=.7)
        elif methods[i] == 'NeuronIntGrad*':
            ax.plot([0, 10, 30, 50, 70, 90], res[i], marker='*', label=r'\textbf{NeuronIntGrad*}',
                    color=tableau20[color_mapping[methods[i]]], linewidth=2, alpha=.7)

    f = open('./roar_auc.txt', 'w+')
    print("########## " + title + " ##########", file=f)
    print("{:<15} {:<10}".format('Method', 'AUC'), file=f)
    print('-' * 20, file=f)
    for k in aucs.keys():
        print("{:<15} {:<10}".format(k, aucs[k]), file=f)
    print("########## " + 'END' + " ##########", file=f)
    f.close()
    plt.rc('text', usetex=True)
    plt.ylim(0, 100)
    plt.xlim(0, 100)
    plt.ylabel('Test Accuracy (%)', fontsize=10)
    plt.xlabel('%of input pixels removed', fontsize=10)
    plt.legend(ncol=3, fontsize=10)
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    # plt.show()
    plt.savefig('./roar_'+dset+'.png', dpi=400)
    plt.close()


def pixel_perturbation_plot(dset, f):
    methods = ['Gradients', 'GBP', 'GradCAM', 'InputMCT', 'InputIntGrad', 'NeuronMCT', 'NeuronIntGrad', 'NeuronIntGrad*']
    res = []
    if dset == 'cifar10':
        gradients = [0.216, 0.285, 0.327, 0.361, 0.390, 0.428, 0.450, 0.477, 0.510, 0.540, 0.751, 0.826, 0.856, 0.864, 0.875, 0.877, 0.863, 0.871]
        res.append(gradients)
        gbp = [0.150, 0.202, 0.242, 0.280, 0.314, 0.333, 0.350, 0.368, 0.381, 0.387, 0.502, 0.591, 0.634, 0.641, 0.641, 0.644, 0.713, 0.831]
        res.append(gbp)
        gradcam = [0.076, 0.092, 0.103, 0.111, 0.120, 0.126, 0.123, 0.134, 0.136, 0.148, 0.192, 0.207, 0.260, 0.311, 0.390, 0.501, 0.643, 0.772]
        res.append(gradcam)
        input_mct = [0.150, 0.168, 0.183, 0.194, 0.200, 0.215, 0.221, 0.229, 0.235, 0.247, 0.368, 0.423, 0.475, 0.517, 0.546, 0.570, 0.636, 0.845]
        res.append(input_mct)
        input_intgrad = [0.139, 0.169, 0.190, 0.207, 0.209, 0.217, 0.219, 0.229, 0.239, 0.246, 0.341, 0.391, 0.429, 0.454, 0.488, 0.510, 0.589, 0.836]
        res.append(input_intgrad)
        neuron_mct = [0.039, 0.058, 0.073, 0.088, 0.100, 0.111, 0.118, 0.125, 0.140, 0.152, 0.222, 0.294, 0.375, 0.466, 0.570, 0.676, 0.765, 0.833]
        res.append(neuron_mct)
        neuron_intgrad = [ 0.039, 0.058, 0.072, 0.088, 0.099, 0.111, 0.118, 0.125, 0.141, 0.150, 0.220, 0.293, 0.377, 0.465, 0.570, 0.667, 0.764, 0.831]
        res.append(neuron_intgrad)
        neuron_intgrad_star = [0.034, 0.047, 0.058, 0.069, 0.079, 0.087, 0.096, 0.101, 0.107, 0.113]
        res.append(neuron_intgrad_star)
        title = "Input Degradation (LeRF) Benchmark on CIFAR10"
    elif dset == 'birdsnap':
        gradients = [0.833, 0.924, 0.953, 0.966, 0.973, 0.975, 0.978, 0.978, 0.979, 0.982]
        res.append(gradients)
        gbp = [0.740, 0.839, 0.881, 0.902, 0.917, 0.931, 0.939, 0.945, 0.953, 0.956]
        res.append(gbp)
        gradcam = [0.079, 0.094, 0.107, 0.125, 0.136, 0.147, 0.149, 0.162, 0.168, 0.174]
        res.append(gradcam)
        input_mct = [0.856, 0.929, 0.954, 0.963, 0.968, 0.972, 0.974, 0.974, 0.975, 0.976]
        res.append(input_mct)
        input_intgrad = [0.852, 0.928, 0.959, 0.966, 0.969, 0.971, 0.973, 0.974, 0.975, 0.976]
        res.append(input_intgrad)
        neuron_mct= [0.061, 0.075, 0.090, 0.103, 0.112, 0.124, 0.137, 0.148, 0.158, 0.166]
        res.append(neuron_mct)
        neuron_intgrad = [0.060, 0.074, 0.089, 0.102, 0.112, 0.124, 0.137, 0.148, 0.158, 0.166]
        res.append(neuron_intgrad)
        neuron_intgrad_star = [0.029, 0.045, 0.059, 0.075, 0.089, 0.102, 0.113, 0.126, 0.137, 0.146]
        res.append(neuron_intgrad_star)
        title = "Input Degradation (LeRF) Benchmark on Birdsnap"
    elif dset == 'imagenet':
        gradients = [0.324, 0.393, 0.446, 0.479, 0.506, 0.528, 0.548, 0.568, 0.583, 0.599]
        res.append(gradients)
        gbp = [0.255, 0.348, 0.405, 0.453, 0.489, 0.517, 0.541, 0.563, 0.580, 0.591]
        res.append(gbp)
        gradcam = [0.065, 0.077, 0.088, 0.100, 0.110, 0.115, 0.117, 0.122, 0.129, 0.132]
        res.append(gradcam)
        input_mct = [0.309, 0.371, 0.408, 0.438, 0.468, 0.489, 0.508, 0.526, 0.541, 0.556]
        res.append(input_mct)
        input_intgrad = [0.305, 0.363, 0.401, 0.431, 0.456, 0.470, 0.487, 0.501, 0.513, 0.527]
        res.append(input_intgrad)
        neuron_mct= [0.050, 0.073, 0.091, 0.100, 0.112, 0.122, 0.131, 0.141, 0.151, 0.157]
        res.append(neuron_mct)
        neuron_intgrad = [0.060, 0.074, 0.089, 0.102, 0.112, 0.124, 0.137, 0.148, 0.158, 0.166]
        res.append(neuron_intgrad)
        neuron_intgrad_star = [0.048,  0.064, 0.076, 0.088, 0.096, 0.106, 0.110, 0.119, 0.126, 0.132]
        res.append(neuron_intgrad_star)
        title = "Input Degradation (LeRF) Benchmark on ImageNet"

    color_mapping = {
        'Gradients': 2,
        'GBP': 4,
        'GradCAM': 6,
        'InputIntGrad': 10,
        'InputMCT': 8,
        'NeuronMCT': 12,
        'NeuronIntGrad': 14,
        'NeuronIntGrad*': 18}

    tableau20 = [(31, 119, 180), (174, 199, 232),
                 (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138),
                 (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213),
                 (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210),
                 (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    aucs = {}
    plt.style.use('seaborn-dark-palette')
    # ax = plt.figure(figsize=(6.5, 3))
    ax = plt.figure(figsize=(5.5, 3))
    # fig, ax = plt.subplots()
    for i in range(len(methods)):
        auc = metrics.auc(np.asarray([0]+list(range(1, 11)))/100, np.asarray([0]+res[i][:10]))
        aucs[methods[i]] = auc
        if i < 5:
            plt.plot([0]+list(range(1, 11)), 100*np.asarray([0]+res[i][:10]), label=methods[i], color=tableau20[color_mapping[methods[i]]], linewidth=2, alpha=.7)
        elif methods[i] == 'NeuronMCT':
            plt.plot([0]+list(range(1, 11)), 100*np.asarray([0]+res[i][:10]), marker='*', label=r'\textbf{NeuronMCT}', color=tableau20[color_mapping[methods[i]]], linewidth=2, alpha=.7)
        elif methods[i] == 'NeuronIntGrad':
            plt.plot([0]+list(range(1, 11)), 100*np.asarray([0]+res[i][:10]), marker='*', label=r'\textbf{NeuronIntGrad}', color=tableau20[color_mapping[methods[i]]], linewidth=2, alpha=.7)
        elif methods[i] == 'NeuronIntGrad*':
            plt.plot([0] + list(range(1, 11)), 100 * np.asarray([0] + res[i][:10]), marker='*',
                     label=r'\textbf{NeuronIntGrad*}', color=tableau20[color_mapping[methods[i]]], linewidth=2, alpha=.7)

    print("########## "+title+" ##########", file=f)
    print("{:<15} {:<10}".format('Method', 'AUC'), file=f)
    print('-'*20, file=f)
    for k in aucs.keys():
        print("{:<15} {:<10}".format(k, aucs[k]), file=f)
    print("########## " + 'END' + " ##########", file=f)
    plt.rc('text', usetex=True)
    plt.ylim(0, 100)
    plt.xlim(0, 11)
    plt.ylabel('Absolute Fractional Output Change (%)', fontsize=8)
    plt.xlabel('%of least important input pixels removed', fontsize=10)
    plt.legend(ncol=3, fontsize=10)
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    # plt.show()
    plt.savefig('./input_degradation_lerf_'+dset+'.png', dpi=400)
    plt.close()

def layerwise(dirs, s):
    plt.style.use('seaborn-dark-palette')
    plt.figure()
    sample = 'samples_layerwise_jaccardsfinal.npy'
    mean = 'layerwise_jaccardsfinal.npy'
    for i, d in enumerate(dirs):
        a = np.load(d+mean)
        sa = np.load(d+sample)
        a = a[1][2]
        stdv = np.std(sa[:, 1, 2], axis=0)
        plt.errorbar(np.arange(len(a)), a, yerr=stdv, fmt='--')
    plt.grid('on')
    plt.xticks(np.arange(len(a)))
    plt.xlabel('Layers')
    plt.ylabel('Jaccard Similarity NeuronIntGrad vs. NeuronMCT')
    plt.legend(['Sparsity'+str(s[i])+'%' for i in range(len(s))], ncol=1, loc='lower right')
    plt.tight_layout(0.05, 0.01)
    plt.savefig('./layerwise_jaccard.png')


def layerwise_all(d, s):
    plt.style.use('seaborn-dark-palette')
    plt.figure(figsize=(6, 3.4))
    sample = 'samples_layerwise_jaccardsfinal.npy'
    mean = 'layerwise_jaccardsfinal.npy'
    a = np.load(d+mean)
    sa = np.load(d+sample)
    b = a[0][1]
    a = a[1][1:]
    a[0] = b
    tmp = a[0].copy()
    tmp2 = a[1].copy()
    a[0], a[1] = tmp2, tmp
    stdv = []
    stdv.append(np.std(sa[:, 0, 1], axis=0))
    stdv.append(np.std(sa[:, 1, 2], axis=0))
    stdv.append(np.std(sa[:, 1, 3], axis=0))
    stdv.append(np.std(sa[:, 1, 4], axis=0))
    stdv.append(np.std(sa[:, 1, 5], axis=0))
    for i in range(len(stdv)):
        sa = a[i]
        sstdv = stdv[i]
        plt.errorbar(np.arange(len(sa)), sa, yerr=sstdv, fmt='--')
    plt.grid('on')
    plt.xticks(np.arange(len(a[0])))
    plt.xlabel('Layers')
    plt.ylabel('Jaccard Similarity NeuronIntGrad vs. All')
    plt.legend(['vs. NeuronMCT', 'vs. DGR(init=1)', 'vs. DGR(init=r)', 'vs. GreedyPruning', 'vs. RandomPruning'], ncol=2, loc='upper left')
    plt.title('Sparsity={}'.format(s))
    plt.tight_layout(0.05, 0.01)
    #plt.show()
    #plt.show()
    plt.savefig('./layerwise_all.png')
    plt.close()


def layerwise_dead(d, s):
    plt.style.use('seaborn-dark-palette')
    plt.figure()
    sample = 'samples_layerwise_deadsfinal.npy'
    mean = 'layerwise_deadsfinal.npy'
    a = np.load(d+mean)
    sa = np.load(d+sample)
    a = np.nan_to_num(a)
    #print(a.shape)
    a_tmp = a.copy()
    a[0], a[1], a[2], a[3], a[4], a[5] = a_tmp[1], a_tmp[2], a_tmp[0], a_tmp[5], a_tmp[3], a_tmp[4]
    stdv = []
    stdv.append(np.std(sa[:, 1], axis=0))
    stdv.append(np.std(sa[:, 2], axis=0))
    stdv.append(np.std(sa[:, 0], axis=0))
    stdv.append(np.std(sa[:, 5], axis=0))
    stdv.append(np.std(sa[:, 3], axis=0))
    stdv.append(np.std(sa[:, 4], axis=0))
    for i in range(len(stdv)):
        sa = a[i]
        sstdv = stdv[i]
        plt.errorbar(np.arange(len(sa)), sa, yerr=sstdv, fmt='--')
    plt.grid('on')
    plt.xticks(np.arange(len(a[0])))
    plt.xlabel('Layers')
    plt.ylabel('FractinDead Neurons of ')
    plt.legend(['NeuronIntGrad', 'vs. NeuronMCT', 'vs. DGR(init=1)', 'vs. DGR(init=r)', 'vs. GreedyPruning', 'vs. RandomPruning'], ncol=1, loc='best')
    plt.title('Sparsity=90')
    plt.tight_layout(0.05, 0.01)
    plt.show()
    plt.savefig('./layerwise_deads.png')
    plt.close()


# layerwise_all('./paths_logs_80/', 80)
# layerwise_all('./paths_logs_90/', 90)
# layerwise_all('./paths_logs_99/', 99)
# make_confusion_matrix(load_mean_paths('./paths_logs_80/jaccardsfinal.npy'), 80)
# make_confusion_matrix(load_mean_paths('./paths_logs_90/jaccardsfinal.npy'), 90)
# make_confusion_matrix(load_mean_paths('./paths_logs_99/jaccardsfinal.npy'), 99)
# make_deads_barplot(np.load('./paths_logs_90/deadsfinal.npy'), 90)
# make_deads_barplot(np.load('./paths_logs_80/deadsfinal.npy'), 80)
# make_deads_barplot(np.load('./paths_logs_99/deadsfinal.npy'), 99)
# roar_plot('cifar10')
# roar_plot('birdsnap')

#f = open('./input_degradation_lerf_auc.txt', 'a+')
#pixel_perturbation_plot('cifar10', f)
#pixel_perturbation_plot('birdsnap', f)
#pixel_perturbation_plot('imagenet', f)
#f.close()
