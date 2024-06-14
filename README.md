# DeltaPhi
Code for "DeltaPhi: Learning Physical Trajectory Residual for PDE Solving".

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

Put `Composites.zip` in `datasets/` and unzip it. The directory structure looks as follows:
```
datasets/Composites/Composites_LBO_basis/Composites_LBO_basis.mat
datasets/Composites/Data/Composites.mat
```

Run the experiment as follows:
``` bash
cd Composites

# The code for NORM (Previous):
python norm.py 
# The code for NORM (DeltaPhi): 
python norm_DeltaPhi.py 
```

### More Cases

Codes for other simulation cases will be released later.



## Implement More Residual Neural Operators

You can create a residual neural operator based on existing direct neural operator by modifying the original `Dataset` and `Model` class.
All other configurations keep unchanged.

1. Dataset. Implement the `Dataset` class which has the `__getitem__()` function returning the original input-output function $(a_i,u_i)$ and the randomly sampled auxiliary sample $(a_{k_i}, u_{k_i}, score_{k_i})$.

2. Model. Concatenate the auxiliary sample $(a_{k_i}, u_{k_i}, score_{k_i})$ with original inputs. Add the auxiliary output function $u_{k_i}$ with the model outputs.


<!-- ## Citations

```

``` -->
