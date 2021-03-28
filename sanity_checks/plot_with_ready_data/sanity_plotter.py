import numpy as np
import matplotlib.pyplot as plt

#this code generates plots using the previously generated data in folders with plots name


def agreg(plot_name, num_plot, ax):
    results = {}
    results['VanillaGradient'] = []
    results['GuidedBackprop'] = []
    results['GradCam'] = []
    results['Grad*Input'] = []
    results['IntegratedGradient'] = []
    results['PruneGrad'] = []
    results['PruneInt'] = []

    color_mapping = {
        'VanillaGradient': 2,
        'GuidedBackprop': 4,
        'GradCam': 6,
        'IntegratedGradient': 10,
        'Grad*Input': 8,
        'PruneGrad': 12,
        'PruneInt': 14
    }

    tableau20 = [(31, 119, 180), (174, 199, 232),
                 (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138),
                 (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213),
                 (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210),
                 (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    AXID = 0
    plt.style.use('seaborn-dark-palette')
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    # fig, ax = plt.subplots(figsize=(6.5, 2.6))
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for k in results.keys():
        y = np.load('./sanity_checks/plot_with_ready_data/ResNet/'+plot_name+'/'+k.replace('*', '_').replace('+', 'P')+str(num_plot)+'-numberOfSamples1000.npy')
        # print(k)
        if k == 'VanillaGradient':
            ax.plot(np.arange(len(y)), y, label=r'Gradients', color=tableau20[color_mapping[k]], linewidth=2, alpha=.7)
        elif k == 'Grad*Input':
            ax.plot(np.arange(len(y)), y, label=r'InputMCT', color=tableau20[color_mapping[k]], linewidth=2, alpha=.7)
        elif k == 'IntegratedGradient':
            ax.plot(np.arange(len(y)), y, label=r'InputIntGrad', color=tableau20[color_mapping[k]], linewidth=2, alpha=.7)
        elif k == 'GradCam':
            ax.plot(np.arange(len(y)), y, label=r'GradCAM', color=tableau20[color_mapping[k]], linewidth=2, alpha=.7)
        elif k == 'GuidedBackprop':
            ax.plot(np.arange(len(y)), y, label=r'GBP', color=tableau20[color_mapping[k]], linewidth=2, alpha=.7)
        elif k == 'PruneGrad':
            ax.plot(np.arange(len(y)), y, marker='*', label=r'\textbf{NeuronMCT}', color=tableau20[color_mapping[k]], linewidth=2, alpha=.7)
        elif k == 'PruneInt':
            ax.plot(np.arange(len(y)), y, marker='*', label=r'\textbf{NeuronIntGrad}', color=tableau20[color_mapping[k]], linewidth=2, alpha=.7)

    plt.rc('text', usetex=True)
    
    plt.ylabel(plot_name, fontsize=10)
    plt.legend(ncol=3, fontsize=10)
    plt.ylim(-1, 1)
    plt.xticks(np.arange(len(y)), ['original', 'linear', 'bottleneck\_15', 'bottleneck\_14', 'bottleneck\_13', 'bottleneck\_12', 'bottleneck\_11', 'bottleneck\_10', 'bottleneck\_9', 'bottleneck\_8', 'bottleneck\_7', 'bottleneck\_6', 'bottleneck\_5', 'bottleneck\_4', 'bottleneck\_3', 'bottleneck\_2', 'bottleneck\_1', 'bottleneck\_0', 'conv1'], rotation=45, fontsize=5)
    
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('./'+str(plot_name)+'.png', format='png', dpi=400)


# agreg('Spearman Rank Correlation ABS', 1, None)
# agreg('Spearman Rank Correlation No ABS', 0, None)
agreg('SSIM', 2, None)
