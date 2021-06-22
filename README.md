
The code here is to reproduce the analysis and experiments of the paper:

Ashkan Khakzar, Soroosh Baselizadeh, Saurabh Khanduja, Christian Rupprecht, Seong Tae Kim, Nassir Navab; **Neural Response Interpretation Through the Lens of Critical Pathways**; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2021, pp. 13528-13538

### Paper
The paper and its supplementary materials are both available on **[CVPR2021 Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Khakzar_Neural_Response_Interpretation_Through_the_Lens_of_Critical_Pathways_CVPR_2021_paper.html)** and on **[arXiv](https://arxiv.org/abs/2103.16886)**. 

### Website
You can visit [The Project's Website](https://camp-explain-ai.github.io/PathwayGrad/) for more details and materials. 

### Citation
Please cite the work using the below BibTeX (also available on the Open Access link above)
``` bash
@InProceedings{Khakzar_2021_CVPR,
   author    = {Khakzar, Ashkan and Baselizadeh, Soroosh and Khanduja, Saurabh and Rupprecht, Christian and Kim, Seong Tae and Navab, Nassir},
   title     = {Neural Response Interpretation Through the Lens of Critical Pathways},
   booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   month     = {June},
   year      = {2021},
   pages     = {13528-13538}
   }
``` 

or this one (available on arXiv):
``` bash
@article{Khakzar2021NeuralRI,
  title={Neural Response Interpretation through the Lens of Critical Pathways},
  author={Ashkan Khakzar and Soroosh Baselizadeh and Saurabh Khanduja and C. Rupprecht and Seong Tae Kim and N. Navab},
  journal={ArXiv},
  year={2021},
  volume={abs/2103.16886}
}
```

## Critical Pathways Analysis
The explained critical pathways extraction methods [NeuronIntGrad, NeuronMCT, Greedy, Random, DGR] are implemented in the `src/Pruner.py` file.

You can find examples of how to use them in `path_analysis.py` file. 
For example, to get the critical pathways by NeuronMCT method:
```
 pruner = Pruner.Pruner(model, data, device)
 pruner.prune_neuron_mct(model_sparsity_threshold, debug=False)
 path = pruner.pruned_activations_mask
 pruner.remove_handles()
```
 
The `path_analysis.py` contains all the analyses for the experiments done to analyze different methods by which we derive critical pathways in the paper.
For Pathway Decoding experiments see `path_decoding/path_decoding.py`. The implementation is based on the Lucent package. 

## Saliency Map Generation (Feature Attribution) via PathwayGrad
Getting saliency maps by using different methods in the paper is mostly alike getting the paths (prev section). 
For example, to get the saliency map of an input based on the NeuronMCT method:
```
 pruner = Pruner.Pruner(model, data, device)
 pruner.prune_neuron_mct(model_sparsity_threshold, debug=False)
 saliency = pruner.generate_saliency(make_single_channel=make_single_channel)
 pruner.remove_handles()
```

### Sanity Checks
The sanity checks have been implemented in PyTorch in the `sanity_checks` folder. See `sanity_checks/sanity_checks.py` for the details. To run, just use this file. 

### ROAR and LeRF
For our implementation of these benchmarks please see [ROAR and LeRF](https://github.com/CAMP-eXplain-AI/RoarTorch). We believe our PyTorch implementation of ROAR can highly benefit the community.

## Questions
Please kindly raise an issue or contact us via email. 
