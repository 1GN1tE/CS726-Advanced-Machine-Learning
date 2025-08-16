import evaluate
import torch 
import numpy as np
import random
import argparse

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from medusa.model.medusa_model_new import MedusaModel

from generate_medusa import MedusaTextGenerator
from task0 import set_seed, get_dataloader, clean_text

import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name","-m", type=str, required=False, default="FasterDecoding/medusa-v1.0-vicuna-7b-v1.5", help="The Huggingface medusa model to be used for inference")
    parser.add_argument("--hf-token","-token", type=str, required=True, help="The Huggingface token for accessing Llama weights")
    parser.add_argument("--use-no-medusa-heads","-nmh", type=int, required=False, default=2, help="The number of medusa heads to be used for inference")
    parser.add_argument("--max-input-len","-mil", type=int, required=False, default=1000, help="Maximum length of the input sequence.")
    parser.add_argument("--max-output-len","-mol", type=int, required=False, default=50, help="Maximum number of new tokens to be generated.")
    parser.add_argument("--beam-width","-w", type=int, required=False, default=2, help="Size of beam width for beam search.")
    parser.add_argument("--decoding-strategy","-ds", type=str, required=False, default="greedy", choices=["single-head", "multi-head"], help="The decoding strategy to be used during inference.")
    parser.add_argument("--debug","-db",type=bool,default=False, help="To print debugging statements.")
    
    args = parser.parse_args() 
    print(args)
    
    set_seed()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MedusaModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    tokenizer = model.get_tokenizer()
    model.eval()
    
    generator = MedusaTextGenerator(model=model, use_no_medusa_heads=args.use_no_medusa_heads, decoding_strategy=args.decoding_strategy, eos_id=tokenizer.eos_token_id, beam_width=args.beam_width, max_output_len=args.max_output_len)
    
    # Load dataset
    dataloader = get_dataloader(tokenizer, args.hf_token, max_input_len=args.max_input_len)
    
    reference_texts = []
    generated_texts = []
    
    total = len(dataloader)
    total_time = 0
    total_generated_tokens = 0
  
    for i, batch in enumerate(dataloader):
        input_prompt, ground_truth = batch 
        
        reference_text = [[tokenizer.decode(tokenizer.encode(out)[:args.max_output_len], skip_special_tokens=True, clean_up_tokenization_spaces=True)] for out in ground_truth][0][0]
        reference_text = clean_text(reference_text)
        reference_texts.append(reference_text)
        
        token_ids = input_prompt['input_ids'].to(device)
        
        start_time = time.time()
        generated_tokens = generator(token_ids)
        end_time = time.time()
        
        generated_text = tokenizer.decode(generated_tokens.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_texts.append(generated_text)
        
        total_time += (end_time - start_time)
        total_generated_tokens += len(generated_tokens)
        
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
    
    rtf = total_time / total_generated_tokens if total_generated_tokens > 0 else float('inf')
    
    print(f"""BLEU: {bleu_score['bleu']}\nROUGE-1: {float(rouge_score['rouge1'])}\nROUGE-2: {float(rouge_score['rouge2'])}\nROUGE-LCS: {float(rouge_score['rougeL'])}\nRTF:{rtf}""")