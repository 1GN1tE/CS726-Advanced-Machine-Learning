import evaluate
import torch 
import numpy as np
import random
import argparse

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from generate import TextGenerator

def set_seed(seed=2024):
    """Sets all relevant random seeds to ensure deterministic results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

    transformers.set_seed(seed)

def clean_text(unprocessed_text):
    return unprocessed_text.lower().strip().translate(str.maketrans({c:None for c in ".,-?':"}))

def get_dataloader(tokenizer, hf_token, max_input_len=512, no_examples=50):
    def collate_fn(batch):
        prompt, completion = zip(*map(lambda ex: [ex['hin_Deva'], ex['eng_Latn']], batch))
        
        source = "Hindi"
        target = "English"
        prefix = f"You are an AI assistant whose purpose is to perform translation. Given the following sentence in {source}, translate it to {target}"
        prompt_with_prompt = [f"{prefix}:\n\n{doc}\n\ncompletion:"  for doc in prompt]
        encoded_prompt = tokenizer.batch_encode_plus(prompt_with_prompt, padding=False, truncation=True, max_length=max_input_len, return_attention_mask=True, return_tensors="pt")
                    
        return encoded_prompt, completion 
    
    dataset = load_dataset("ai4bharat/IN22-Gen", split="test", token=hf_token)
    dataset = dataset.select(range(no_examples))
    dataset.set_format(type='torch', columns=['eng_Latn', 'hin_Deva'])
    
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    return dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token","-token", type=str, required=True, help="The Huggingface token for accessing Llama weights")
    parser.add_argument("--model-name","-m", type=str, required=False, default="meta-llama/Llama-2-7b-hf", help="The Huggingface model to be used for inference")
    parser.add_argument("--max-input-len","-mil", type=int, required=False, default=1000, help="Maximum length of the input sequence.")
    parser.add_argument("--max-output-len","-mol", type=int, required=False, default=50, help="Maximum number of new tokens to be generated.")
    parser.add_argument("--decoding-strategy","-ds", type=str, required=False, default="greedy", choices=["greedy", "random", "topk", "nucleus"], help="The decoding strategy to be used during inference.")
    parser.add_argument("--tau","-t", type=int, required=False, default=1, help="Temperature value to be used with Random Sampling.")
    parser.add_argument("--k","-k", type=int, required=False, default=10, help="k value to be used in Top-k and Nucleus sampling.")
    parser.add_argument("--p","-p", type=int, required=False, default=0.9, help="p value to be used in Nucleus sampling.")
    parser.add_argument("--debug","-db",type=bool,default=False, help="To print debugging statements.")
    
    args = parser.parse_args() 
    print(args)
    
    set_seed(2025)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=args.hf_token)
    tokenizer.pad_token = tokenizer.eos_token    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, token=args.hf_token).to(device)
    model.eval()
    generator = TextGenerator(model=model, decoding_strategy=args.decoding_strategy, eos_id=tokenizer.eos_token_id, max_output_len=args.max_output_len, tau=args.tau, k=args.k, p=args.p)
    
    # Load dataset
    dataloader = get_dataloader(tokenizer, args.hf_token, max_input_len=args.max_input_len)
    
    reference_texts = []
    generated_texts = []
  
    total = len(dataloader)
    for i, batch in enumerate(dataloader):
        input_prompt, ground_truth = batch 
        
        # import pdb; pdb.set_trace()
        reference_text = [[tokenizer.decode(tokenizer.encode(out)[:args.max_output_len], skip_special_tokens=True, clean_up_tokenization_spaces=True)] for out in ground_truth][0][0]
        reference_text = clean_text(reference_text)
        reference_texts.append(reference_text)
        
        token_ids = input_prompt['input_ids'].to(device)
        
        generated_tokens = generator(token_ids)
        generated_text = tokenizer.decode(generated_tokens.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_text = clean_text(generated_text)
        generated_texts.append(generated_text)
        
        if args.debug:
            print(f'Example: {i+1}/{total}')
            print(f'Input Prompt:', tokenizer.decode(input_prompt['input_ids'][0]))
            print('Reference:', reference_texts[-1])
            print('Ground Truth:', generated_texts[-1])
            print()
            print()
    
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    
    bleu_score = bleu.compute(predictions=generated_texts, references=reference_texts, max_order=1)
    rouge_score = rouge.compute(predictions=generated_texts, references=reference_texts)
    
    print(f"""BLEU: {bleu_score['bleu']}\nROUGE-1: {float(rouge_score['rouge1'])}\nROUGE-2: {float(rouge_score['rouge2'])}\nROUGE-LCS: {float(rouge_score['rougeL'])}""")
        
        