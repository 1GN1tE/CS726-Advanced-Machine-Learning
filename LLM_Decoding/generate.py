import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        max_output_len: int = 10,
        tau: int = 1,
        k: int = 10,
        p: int = 0.5
    ) -> None:
        '''
            Initialize the TextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            tau: Temperature parameter for random sampling
            k: Top-k parameter for top-k sampling
            p: Cumulative probability threshold for nucleus sampling
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tau = tau
        self.k = k 
        self.p = p
        
        if decoding_strategy == "greedy":
            self.generator_func = self.greedy_decoding
        elif decoding_strategy == "random":
            self.generator_func = self.random_sampling
        elif decoding_strategy == "topk":
            self.generator_func = self.topk_sampling
        elif decoding_strategy == "nucleus":
            self.generator_func = self.nucleus_sampling

    def __call__(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"], 
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def greedy_decoding(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]: 
        '''
            Implement Greedy decoding technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        original_len = input_ids.shape[1]
        
        for _ in range(self.max_output_len):
            
            predicted = self.model(input_ids)
            # print(predicted)
            
            next_token_logits = predicted.logits[0, -1, :]
            # print(next_token_logits)
            
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            # print(next_token_id)
            
            if next_token_id.item() == self.eos_token_id:
                break
            
            next_token_id = next_token_id.unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
        
        return input_ids[0, original_len:]
        
    def random_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Random sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        original_len = input_ids.shape[1]
        
        for _ in range(self.max_output_len):
            
            predicted = self.model(input_ids)
            # print(predicted)
            
            next_token_logits = predicted.logits[0, -1, :]
            # print(next_token_logits)
            
            scaled_logits = next_token_logits / self.tau
            probs = torch.softmax(scaled_logits, dim=-1)
            
            next_token_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            # print(next_token_id)
            
            if next_token_id.item() == self.eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
        
        return input_ids[0, original_len:]
    
    def topk_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Top-k sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        original_len = input_ids.shape[1]
        
        for _ in range(self.max_output_len):
            
            predicted = self.model(input_ids)
            next_token_logits = predicted.logits[0, -1, :]
            
            topk_logits, topk_indices = torch.topk(next_token_logits, self.k)
            
            topk_probs = torch.softmax(topk_logits, dim=-1)
            
            next_token_pos = torch.multinomial(topk_probs, num_samples=1)
            next_token_id = topk_indices[next_token_pos].unsqueeze(0)
            
            if next_token_id.item() == self.eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
        return input_ids[0, original_len:]
    
    def nucleus_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Nucleus sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        original_len = input_ids.shape[1]
        
        for _ in range(self.max_output_len):
            
            predicted = self.model(input_ids)
            next_token_logits = predicted.logits[0, -1, :]
            
            probs = torch.softmax(next_token_logits, dim=-1)
            
            sorted_prob, sorted_indices = torch.sort(probs, descending=True)
            cummilative_probs = torch.cumsum(sorted_prob, dim=-1)
            
            cutoff = torch.where(cummilative_probs > self.p)[0]
            # print(cutoff)
            if len(cutoff) == 0:
                chosen_logits = probs.size(0)
            else :
                chosen_logits = cutoff[0].item() + 1
            
            top_p_probs = sorted_prob[:chosen_logits]
            top_p_indices = sorted_indices[:chosen_logits]
            
            top_p_probs = top_p_probs / top_p_probs.sum()
            
            next_token_pos = torch.multinomial(top_p_probs, num_samples=1)
            next_token_id = top_p_indices[next_token_pos].unsqueeze(0)
            
            if next_token_id.item() == self.eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
        return input_ids[0, original_len:]