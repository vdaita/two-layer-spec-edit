from transformers.generation.stopping_criteria import StoppingCriteria
from transformers.generation.candidate_generator import (
    CandidateGenerator,
    _crop_past_key_values,
)
import torch
from typing import Optional, Tuple
from config import Config

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
        draft_model,
        config: Config,
        num_pld_tokens=20,
        num_runs=4,
    ):
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.draft_model.generation_config.pad_token_id = tokenizer.pad_token_id
        self.config = config
        self.past_key_values = None
        self.num_runs = num_runs
        self.num_pld_tokens = num_pld_tokens

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
                prompt_lookup_num_tokens=self.num_pld_tokens,
                max_new_tokens=1000,
                stopping_criteria=stopping_criteria,
                past_key_values=self.past_key_values,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=self.config.temperature,
                do_sample=False
            )
        else:
            generation = self.draft_model.generate(
                inputs=input_ids,
                attention_mask=torch.ones(
                    input_ids.shape[-1], device=input_ids.device
                ).unsqueeze(0),
                prompt_lookup_num_tokens=self.num_pld_tokens,
                max_matching_ngram_size=self.config.max_matching_ngram_size,
                max_new_tokens=1000,
                stopping_criteria=stopping_criteria,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=self.config.temperature,
                do_sample=False
            )

        input_ids = input_ids.to(old_device)

        self.pred_tokens_count = generation.sequences.shape[-1] - input_ids.shape[-1]
        self.past_key_values = generation.past_key_values

        return generation.sequences, None

    def update_candidate_strategy(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int
    ):
        pass