from transformers.generation.stopping_criteria import StoppingCriteria
from transformers.generation.candidate_generator import (
    CandidateGenerator,
    _crop_past_key_values,
)
import torch
from typing import Optional, Tuple
from config import *

class NumRunsStoppingCriteria(StoppingCriteria):
    def __init__(self, max_num_runs=4):
        self.max_num_runs = max_num_runs
        self.num_runs = 0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor:
        self.num_runs += 1
        return self.num_runs >= self.max_num_runs
    

class TwoLayerLookupCandidateGenerator(CandidateGenerator):
    def __init__(
        self,
        tokenizer,
        prompt_tokens,
        draft_model,
        input_ids,
        code_ids,
        num_runs=4,
        **diff_prompt_args,
    ):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens
        self.draft_model = draft_model
        self.input_ids = input_ids
        self.code_ids = code_ids
        self.draft_model.generation_config.pad_token_id = tokenizer.pad_token_id
        self.past_key_values = None
        self.num_runs = num_runs
        self.start_token_index = self.input_ids.shape[-1]

    def get_candidates(
        self, input_ids: torch.LongTensor
    ) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        if self.past_key_values:
            self.past_key_values = _crop_past_key_values(
                self.draft_model, self.past_key_values, input_ids.shape[-1] - 1
            )

        stopping_criteria = [
            NumRunsStoppingCriteria(self.num_runs),
        ]
        old_device = input_ids.device
        input_ids = input_ids.to(self.draft_model.device)

        if self.past_key_values:
            generation = self.draft_model.generate(
                inputs=input_ids,
                attention_mask=torch.ones(
                    input_ids.shape[-1], device=input_ids.device
                ).unsqueeze(0),
                prompt_lookup_num_tokens=self.prompt_tokens,
                max_new_tokens=1000,
                stopping_criteria=stopping_criteria,
                past_key_values=self.past_key_values,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=temperature,
            )
        else:
            generation = self.draft_model.generate(
                inputs=input_ids,
                attention_mask=torch.ones(
                    input_ids.shape[-1], device=input_ids.device
                ).unsqueeze(0),
                prompt_lookup_num_tokens=self.prompt_tokens,
                max_matching_ngram_size=max_matching_ngram_size,
                max_new_tokens=1000,
                stopping_criteria=stopping_criteria,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=temperature,
            )

        input_ids = input_ids.to(old_device)

        self.pred_tokens_count = generation.sequences.shape[-1] - input_ids.shape[-1]
        self.past_key_values = generation.past_key_values
        self.past_top_scores = (
            torch.stack(generation.scores, dim=1).max(dim=1).values[0]
        )

        return generation.sequences, torch.stack(generation.scores, dim=1)

    def update_candidate_strategy(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int
    ):
        pass