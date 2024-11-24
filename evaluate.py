#!/usr/bin/env python
# SPDX-FileCopyrightText: (c) iSE UIUC Research Group
#
# SPDX-License-Identifier: Apache-2.0

import os

# coding: utf-8
import time

import fire
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import load_config_from_json
from two_layer_candidate_generator import TwoLayerLookupCandidateGenerator
from utils import _get_default_candidate_generator_generator, print_update, save_file

# Example shot for the model
shot = """## Code Before:
def add(a, b):
    return a + b
## Instruction:
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
## Code After:
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y"""


def run_evaluation(config_file: str):
    # Load configuration from JSON file
    config = load_config_from_json(config_file)

    # Load the draft model
    draft_model = AutoModelForCausalLM.from_pretrained(
        config.draft_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
        device_map="auto",
    )

    # Load the tokenizer and main model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
        device_map="auto",
    )

    # Load the dataset
    ds = load_dataset(config.dataset_name, split=config.dataset_split)

    stats = []
    regular_get_candidate_generator = model._get_candidate_generator
    tokenizer.eos_token_id = (
        tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
    )

    # Iterate over the dataset
    for row_idx, row in tqdm(enumerate(ds)):
        output = {}
        # Prepare the input text for the model
        input_text = (
            shot
            + "\n## Code Before:\n{code_text}\n## Instruction: {question}\n## Code After:\n".format(
                code_text=row["code"], question=row["change_request"]
            )
        )
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": input_text},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        # Tokenize the code
        code_tokens = tokenizer(row["code"], return_tensors="pt").to(model.device)
        starting_input_tokens = inputs.shape[-1]
        max_new_tokens = code_tokens.input_ids.shape[-1] + 512

        # Assisted decoding
        if config.USE_ASSISTED_DECODING:
            model._get_candidate_generator = (regular_get_candidate_generator).__get__(
                model, type(model)
            )
            start_time = time.perf_counter()
            assisted_decoding_output = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                assistant_model=draft_model,
                use_cache=True,
                temperature=config.temperature,
                do_sample=False,
            )
            end_time = time.perf_counter()
            output["assisted_decoding"] = end_time - start_time

        # Regular decoding
        if config.USE_REGULAR_DECODING:
            model._get_candidate_generator = (regular_get_candidate_generator).__get__(
                model, type(model)
            )
            start_time = time.perf_counter()
            assisted_decoding_output = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
                temperature=config.temperature,
                do_sample=False,
            )
            end_time = time.perf_counter()
            output["regular_decoding"] = end_time - start_time

        # Iterate over lookup tokens
        for lt in config.lookup_tokens:
            # Using HuggingFace Prompt Lookup Decoding
            if config.USE_PROMPT_LOOKUP_DECODING:
                model._get_candidate_generator = (
                    regular_get_candidate_generator
                ).__get__(model, type(model))
                start_time = time.perf_counter()
                pld_output = model.generate(
                    input_ids=inputs,
                    max_new_tokens=max_new_tokens,
                    max_matching_ngram_size=config.max_matching_ngram_size,
                    return_dict_in_generate=True,
                    output_scores=True,
                    prompt_lookup_num_tokens=lt,
                    use_cache=True,
                    temperature=config.temperature,
                    attention_mask=torch.ones(
                        inputs.shape[-1], device=inputs.device
                    ).unsqueeze(0),
                    do_sample=False,
                )
                end_time = time.perf_counter()

                output[f"pld_{lt}"] = end_time - start_time
                output[f"pld_{lt}_text"] = tokenizer.decode(
                    pld_output.sequences[0, starting_input_tokens:]
                )

            # Iterate over model draft tokens
            for mdt in config.model_draft_tokens:
                # Using Two Layer Lookup Decoding
                if config.USE_TWO_LAYER_LOOKUP_DECODING:
                    two_layer_candidate_generator = TwoLayerLookupCandidateGenerator(
                        tokenizer,
                        draft_model,
                        config,
                        num_pld_tokens=lt,
                        num_runs=mdt,
                    )
                    model._get_candidate_generator = (
                        _get_default_candidate_generator_generator(
                            two_layer_candidate_generator
                        )
                    ).__get__(model, type(model))

                    start_time = time.perf_counter()
                    two_layer_out = model.generate(
                        inputs=inputs,
                        prompt_lookup_num_tokens=1,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        temperature=config.temperature,
                        return_dict_in_generate=True,
                        do_sample=False,
                    )
                    end_time = time.perf_counter()
                    output[f"method_{lt}_{mdt}"] = end_time - start_time
                    output[f"method_{lt}_{mdt}_text"] = tokenizer.decode(
                        two_layer_out.sequences[0, starting_input_tokens:]
                    )

        # Append the output to stats and print update
        stats.append(output)
        print_update(output)
        save_file(stats, config.output_file)

    # Save the final stats to the output file
    save_file(stats, config.output_file)


if __name__ == "__main__":
    fire.Fire(run_evaluation)
