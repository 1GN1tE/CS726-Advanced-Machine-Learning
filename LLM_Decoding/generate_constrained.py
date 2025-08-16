import torch
import torch.nn as nn
import warnings

from jaxtyping import Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TokenTrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word_idx = None

def insert_tokens_in_trie(root: TokenTrieNode, token_ids: List[int], word_idx: int):
    current_node = root
    
    for tid in token_ids:
        if tid not in current_node.children:
            current_node.children[tid] = TokenTrieNode()
        current_node = current_node.children[tid]
    
    current_node.is_end_of_word = True
    current_node.word_idx = word_idx

def build_token_trie(tokenizer, word_list: List[str]):
    trie_roots = []
    tokenized_words = []
    
    for idx, w in enumerate(word_list):
        w_stripped = w.strip()
        if not w_stripped:
            trie_roots.append(None)
            tokenized_words.append([])
            continue
        
        token_ids = tokenizer.encode(w_stripped, add_special_tokens=False)
        root = TokenTrieNode()
        insert_tokens_in_trie(root, token_ids, idx)
        trie_roots.append(root)
        tokenized_words.append(token_ids)
    
    return trie_roots, tokenized_words

class ConstrainedTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        eos_id: int, 
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the ConstrainedTextGenerator class.
            
            model: LLM
            tokenizer: LLM's tokenizer.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Word-Constrained decoding technique. (refer assignment document for more details)
            
            `word_list`: contains bag of words for the particular example

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
        trie_roots, _ = build_token_trie(self.tokenizer, word_list)
        word_set = set(range(len(word_list)))
        
        trie_positions = [root for root in trie_roots]
        
        batch_size = input_ids.size(0)
        assert batch_size == 1, "batch_size=1 allowed"
        original_length = input_ids.size(1)
        
        current_ids = input_ids.clone()
        
        for step_idx in range(self.max_output_len):
            outputs = self.model(
                current_ids,
                use_cache=False,
                return_dict=True
            )
            # shape = (1, seq_len, vocab_size)
            logits = outputs.logits[:, -1, :]  # shape: (1, vocab_size)

            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            next_token_id = None
            found_feasible = False
            
            # Are all words used?
            all_used = (len(word_set) == 0)
            
            vocab_size = sorted_indices.size(1)
            
            for i in range(vocab_size):
                candidate_int = sorted_indices[0, i].item()
                
                # skip EOS if not all words are used
                if not all_used and candidate_int == self.eos_token_id:
                    continue
                
                # If all words are used
                if all_used:
                    next_token_id = candidate_int
                    found_feasible = True
                    break
                
                # Otherwise, check if this candidate can match
                # at least one unmatched word's partial trie or root
                some_ok = False
                for w_idx in list(word_set):
                    node = trie_positions[w_idx]
                    if node is None:
                        # empty word
                        some_ok = True
                        break
                    
                    if candidate_int in node.children:
                        some_ok = True
                        break
                    else:
                        root_node = trie_roots[w_idx]
                        if root_node and (candidate_int in root_node.children):
                            some_ok = True
                            break
                
                if some_ok:
                    next_token_id = candidate_int
                    found_feasible = True
                    break
            
            # If no feasible candidate found, fallback
            if not found_feasible:
                next_token_id = sorted_indices[0, 0].item()
            
            # Build 2D tensor for the chosen token
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=logits.device)
            current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
            
            # Update partial matching
            for w_idx in list(word_set):
                node = trie_positions[w_idx]
                if node is None:
                    continue
                
                if next_token_id in node.children:
                    node = node.children[next_token_id]
                else:
                    root_node = trie_roots[w_idx]
                    if root_node and (next_token_id in root_node.children):
                        node = root_node.children[next_token_id]
                    else:
                        node = trie_roots[w_idx]
                
                trie_positions[w_idx] = node
                if node and node.is_end_of_word:
                    # matched the entire word
                    word_set.remove(w_idx)
            
            # If we picked EOS and no words remain unmatched => stop
            if next_token_id == self.eos_token_id and len(word_set) == 0:
                break
        
        return current_ids[0, original_length:]
