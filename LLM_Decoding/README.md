## Programming Assignment 3: LLM sampling and decoding techniques

### Setting up the codebase [very crucial]
1. Getting access to Llama weights: https://www.llama.com/docs/getting-the-models/hugging-face/ (Get access to `meta-llama/Llama-2-7b-hf`)
2. Getting access to IN22 dataset: Sign in to your huggingface account and accept the agreement here: https://huggingface.co/datasets/ai4bharat/IN22-Gen
3. Install dependencies: The codebase has a `environment.yml` file, which you can use to create a new envrinoment as follows:
```bash 
conda env create -f environment.yml
```
4. Install Medusa: Next, you need Medusa's codebase which can be installed as follows:
```bash
conda activate cs726_a3
git clone https://github.com/Darshan7575/Medusa.git
cd Medusa
pip install -e .
```
4. Creating Huggingface access token: Finally, you need to create huggingface token to access the model weights and dataset. Follow these instructions: https://huggingface.co/docs/hub/en/security-tokens


### How to run?
#### Task 0

1. **Greedy Decoding**: To run the code on a specific GPU, use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token "<your_hf_token>" --decoding-strategy "greedy"
```
To run the code on CPU, use the following command (*please note that this will be extremely slow*):
```bash
CUDA_VISIBLE_DEVICES=-1 python task0.py --hf-token "<your_hf_token>" --decoding-strategy "greedy"
```
Additionally, if you want to check the input, reference text and your predicted outputs, you can use:
```bash
CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token "<your_hf_token>" --decoding-strategy "greedy" --debug true
```

2. **Random Sampling**: Follow the same steps as above, but with additional arguments
```bash
CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token "<your_hf_token>" --decoding-strategy "random" --tau <tau value>
```

3. **Top-k Sampling**: Follow the same steps as above, but with additional arguments
```bash
CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token "<your_hf_token>" --decoding-strategy "topk" --k <k value>
```

4. Nucleus Sampling: Follow the same steps as above, but with additional arguments
```bash
CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token "<your_hf_token>" --decoding-strategy "nucleus" --p <p value>
```

#### Task 1
Similar to the previous task, you can run the script as follows:
```bash
CUDA_VISIBLE_DEVICES=0 python task1.py --hf-token "<your_hf_token>" --word_list <path to the word_lists.txt file>
```

#### Task 2
Similar to previous task, you can run the script as follows:
1. **Single head decoding**
```bash 
CUDA_VISIBLE_DEVICES=0 python task2.py --hf-token "<your_hf_token>" --decoding-strategy "single-head"
```

2. **Multiple head decoding**
```bash
CUDA_VISIBLE_DEVICES=0 python task2.py --hf-token "<your_hf_token>" --decoding-strategy "multi-head" --beam-width <beam width> --use-no-medusa-heads <no of medusa heads to be used>
```