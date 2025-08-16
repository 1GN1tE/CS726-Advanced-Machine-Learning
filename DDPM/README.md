# Diffusion-and-Guidance

You need to edit just ddpm.py file without changing the function names and signatues of functions outside `__main__`. You can add more options to the argparser as required for your experiments. For example, class labels and guidance scales for part 2.

To setup the environment, follow these steps:

```
conda create --name cs726env python=3.8 -y
conda activate cs726env
```
Install the dependencies
```
pip install -r requirements.txt
```
To install torch, you can follow the steps [here](https://pytorch.org/get-started/locally/). You'll need to know the cuda version on the server. Use `nvitop` command to know the version first. If you have cuda version 12.4, you can just do:

```
pip install torch
```

In case multiple GPUs are present in the system, we recommend using the environment variable `CUDA_VISIBLE_DEVICES` when running your scripts. For example, below command ensures that your script runs on 7th GPU. 

```
CUDA_VISIBLE_DEVICES=7 python ddpm.py --mode train --dataset moons
```

CUDA error messages can often be cryptic and difficult to debug. In such cases, the following command can be quite useful:
```
CUDA_VISIBLE_DEVICE=-1 python ddpm.py --mode train --dataset moons
```
This forces the script to run exclusively on the CPU.





