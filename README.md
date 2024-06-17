# DeltaPhi
Code for "[DeltaPhi: Learning Physical Trajectory Residual for PDE Solving](https://arxiv.org/pdf/2406.09795)".

## Installing Environment

Create and activate an Anaconda Environment:
```
conda create -n DeltaPhi python=3.9
conda activate DeltaPhi
```

Install Pytorch with following commands (refer to  [Other Versions](https://pytorch.org/get-started/previous-versions/) for additional cuda versions.):
```
# for CUDA 11.8
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# for CUDA 12.1
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install additional required packages with following commands:
```
pip install -r requirement.txt
```


## Running Experiments

### Case 1: Composites

Download the dataset from [Google Driver](https://drive.google.com/file/d/1BNCxwwI3M4OUg3sC-8LljLT98joB0G9D/view?usp=sharing).
Put `Composites.zip` in `datasets/` and unzip it. 
``` bash
cd datasets
unzip Composites.zip
```

The directory structure is as follows:
```
datasets/Composites/Composites_LBO_basis/Composites_LBO_basis.mat
datasets/Composites/Data/Composites.mat
```

Run the experiment using the following scripts:
``` bash
cd Composites

# The code for NORM (Previous):
CUDA_VISIBLE_DEVICES=0 python norm.py 
# The code for NORM (DeltaPhi): 
CUDA_VISIBLE_DEVICES=0 python norm_DeltaPhi.py 
```

### Case 2: HeatTransfer

Download the dataset from [Google Driver](https://drive.google.com/file/d/1U9cWx7AZzmzF268SDOyRdubLaSEKNAtx/view?usp=sharing).
Put `HeatTransfer.zip` in `datasets/` and unzip it. 
``` bash
cd datasets
unzip HeatTransfer.zip
```

The directory structure is as follows:
```
datasets/HeatTransfer/Data/HeatTransfer.mat
datasets/HeatTransfer/HeatTransfer_LBO_basis/lbe_ev_input.mat
datasets/HeatTransfer/HeatTransfer_LBO_basis/lbe_ev_output.mat
```

Run the experiment using the following scripts:
``` bash
cd HeatTransfer

# The code for NORM (Previous):
CUDA_VISIBLE_DEVICES=0 python norm.py 
# The code for NORM (DeltaPhi): 
CUDA_VISIBLE_DEVICES=0 python norm_DeltaPhi.py 
```

### More Cases

Codes for additional simulation cases will be released later.



## Create More Residual Neural Operators

You can create a residual neural operator based on an existing direct neural operator by modifying the original `Dataset` and `Model` class.
All other configurations keep unchanged.

1. Dataset. Implement the `Dataset` class in which the `__getitem__()` function returns not only the original input-output function $(a_i,u_i)$, but also the randomly sampled auxiliary sample $(a_{k_i}, u_{k_i}, score_{k_i})$.

2. Model. Modify the `Model` class in two aspects: (a) Auxiliary sample integration: Concatenate original inputs with auxiliary sample $(a_{k_i}, u_{k_i}, score_{k_i})$. (b) Trajectory residual connection: Take the summation of the auxiliary output function $u_{k_i}$ and original model outputs, as the final model outputs.



## Acknowledgement

We greatly appreciate the following open-source works, as they form the foundation of our work.

- https://github.com/gengxiangc/NORM
- https://github.com/neuraloperator/neuraloperator
- https://github.com/HaoZhongkai/GNOT
- https://github.com/alasdairtran/fourierflow
- https://github.com/pdearena/pdearena
- https://github.com/scaomath/galerkin-transformer
- https://github.com/lu-group/mionet


<!-- ## Citations

```

``` -->