#conda create -y -n prunegrad python=3.7
#conda activate prunegrad  # Conda activate does not work from script(subshell)
pip install texttable jupyter tqdm
conda install -y pytorch=1.2.0 torchvision=0.4.0 cudatoolkit=10.0.130 -c pytorch
conda install -y matplotlib
conda install -y -c conda-forge scikit-image
conda install -y pandas  # Needed for NIH-ChestXrays dataset
pip install scikit-multilearn  # Stratification of NIH dataset for mul
conda install pillow=6.2.1