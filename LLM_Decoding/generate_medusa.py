import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List
from copy import deepcopy

warnings.filterwarnings("ignore")

class MedusaTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the MedusaTextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            use_no_medusa_heads: Number of medusa heads to be used (maximum:5) (denoted as S).
            beam_width: Maximum number of candidates that can be present in the beam (denoted as W).
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        
        assert use_no_medusa_heads <= 5, "The current medusa model supports at max 5 heads"
        self.no_heads = use_no_medusa_heads + 1
        
        if decoding_strategy == "single-head":
            self.generator_func = self.single_head_decoding
        elif decoding_strategy == "multi-head":
            self.generator_func = self.multi_head_decoding
        
    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def single_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement Single-head decoding technique. Use only LM head for decoding here (refer assignment document for more details)

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
        original_length = input_ids.shape[1]

        for _ in range(self.max_output_len):
            with torch.no_grad():
                predicted = self.model(input_ids)  # returns logits in predicted.logits

            next_token_logits = predicted.logits[0, -1, :]  # Last position
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            if next_token_id.item() == self.eos_token_id:
                break

            # Append next token
            next_token_id = next_token_id.unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

        return input_ids[0, original_length:]

    def multi_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement multi-head decoding technique. (refer assignment document for more details)

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
        device = input_ids.device
        original_length = input_ids.size(1)
        max_new_tokens = self.max_output_len

        full_sequence = input_ids.clone()
        generated_count = 0

        while generated_count < max_new_tokens:
            # Forward pass to get LM + Medusa heads for *last* position
            with torch.no_grad():
                medusa_logits, _, lm_logits = self.model(
                    full_sequence,
                    medusa_forward=True,
                    output_orig=True,
                    use_cache=True
                )

            # p_t => LM distribution at the last time-step
            p_lm = torch.log_softmax(lm_logits[0, -1, :], dim=-1)
            
            # p_{t+1}..p_{t+S} => Medusa head distributions
            # shape: [S, 1, seq_len, vocab]
            p_medusa_list = [
                torch.log_softmax(medusa_logits[i, 0, -1, :], dim=-1)
                for i in range(self.no_heads - 1)  # i in [0..S-1]
            ]

            # Combine them in a list => [p_lm, p_{t+1}, ..., p_{t+S}]
            p_distributions = [p_lm] + p_medusa_list

            # "Chunked" beam search to produce S+1 tokens
            beam = [( [], 0.0 )]

            for dist in p_distributions:
                new_beam = []

                for (tok_seq, partial_score) in beam:
                    # If the last chosen token is EOS, we carry it forward unexpanded
                    if tok_seq and tok_seq[-1] == self.eos_token_id:
                        new_beam.append((tok_seq, partial_score))
                        continue

                    # Top-k from dist (log-probs) directly
                    topk_logprobs, topk_indices = torch.topk(dist, self.beam_width, dim=-1)

                    for i_k in range(self.beam_width):
                        cand_token_id = topk_indices[i_k].item()
                        cand_logp = topk_logprobs[i_k].item()

                        new_seq = tok_seq + [cand_token_id]
                        new_score = partial_score + cand_logp
                        new_beam.append((new_seq, new_score))

                # Prune to top beam_width
                new_beam.sort(key=lambda x: x[1], reverse=True)
                beam = new_beam[:self.beam_width]

            # Pick the single best expansion by summing LM-head log-probs for those new tokens
            best_expansion = None
            best_lm_score = float("-inf")

            for (new_tokens_list, _) in beam:
                if not new_tokens_list:
                    continue

                # Sum LM log probs over these new tokens
                lm_score = self._compute_lm_score_for_chunk(full_sequence, new_tokens_list)
                if lm_score > best_lm_score:
                    best_lm_score = lm_score
                    best_expansion = new_tokens_list

            if best_expansion is None:
                # Means beam was empty or something unexpected
                break

            # Append best_expansion to full_sequence
            best_expansion_ids = torch.tensor(best_expansion, dtype=torch.long, device=device).unsqueeze(0)
            full_sequence = torch.cat([full_sequence, best_expansion_ids], dim=1)

            # Stop if EOS is generated
            if self.eos_token_id in best_expansion:
                break

            generated_count += len(best_expansion)

        # Return newly generated portion only
        return full_sequence[0, original_length:]

    def _compute_lm_score_for_chunk(self, prefix_ids: torch.Tensor, new_tokens: List[int]) -> float:
        """
        Computes sum of LM-head log probabilities for `new_tokens`,
        given an existing prefix `prefix_ids`.
        """
        device = prefix_ids.device
        running_input = prefix_ids.clone()
        total_score = 0.0

        for tok in new_tokens:
            with torch.no_grad():
                out = self.model(running_input)
            lm_dist = torch.log_softmax(out.logits[0, -1, :], dim=-1)
            total_score += lm_dist[tok].item()

            # Append this token
            tok_tensor = torch.tensor([[tok]], dtype=torch.long, device=device)
            running_input = torch.cat([running_input, tok_tensor], dim=1)

            # Early stop if we hit EOS
            if tok == self.eos_token_id:
                break

        return total_score
