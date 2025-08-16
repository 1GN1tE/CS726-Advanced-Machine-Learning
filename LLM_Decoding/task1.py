import evaluate
import torch 
import argparse

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from generate_constrained import ConstrainedTextGenerator
from task0 import get_dataloader, set_seed, clean_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token","-token", type=str, required=True, help="The Huggingface token for accessing Llama weights")
    parser.add_argument("--word-list","-wl", type=str, required=False, default="word_lists.txt", help="The path of the word list for constrained decoding")
    parser.add_argument("--model-name","-m", type=str, required=False, default="meta-llama/Llama-2-7b-chat-hf", help="The Huggingface model to be used for inference")
    parser.add_argument("--max-input-len","-mil", type=int, required=False, default=1000, help="Maximum length of the input sequence.")
    parser.add_argument("--max-output-len","-mol", type=int, required=False, default=50, help="Maximum number of new tokens to be generated.")
    parser.add_argument("--debug","-db",type=bool,default=False, help="To print debugging statements.")
    
    args = parser.parse_args() 
    print(args)
    
    set_seed()
    
    with open(args.word_list, 'r') as f:
        word_lists = [line.strip() for line in f]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=args.hf_token)
    tokenizer.pad_token = tokenizer.eos_token    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, token=args.hf_token).to(device)
    model.eval()
    generator = ConstrainedTextGenerator(model=model, tokenizer=tokenizer, eos_id=tokenizer.eos_token_id, max_output_len=args.max_output_len)
    
    # Load dataset
    dataloader = get_dataloader(tokenizer, args.hf_token, max_input_len=args.max_input_len)
    
    reference_texts = []
    generated_texts = []
    
    total = len(dataloader)
  
    for i, batch in enumerate(dataloader):
        input_prompt, ground_truth = batch 
        
        reference_text = [[tokenizer.decode(tokenizer.encode(out)[:args.max_output_len], skip_special_tokens=True, clean_up_tokenization_spaces=True)] for out in ground_truth][0][0]
        reference_text = clean_text(reference_text)
        reference_texts.append(reference_text)
        
        token_ids = input_prompt['input_ids'].to(device)
        word_list = word_lists[i].split('\t')
        
        generated_tokens = generator(token_ids, word_list=word_list)
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
        
        