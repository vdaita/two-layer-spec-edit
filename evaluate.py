#!/usr/bin/env python
# SPDX-FileCopyrightText: (c) iSE UIUC Research Group
#
# SPDX-License-Identifier: Apache-2.0

# coding: utf-8

import json
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import (
    CandidateGenerator,
    _crop_past_key_values,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteria

from datasets import load_dataset
from two_layer_candidate_generator import TwoLayerLookupCandidateGenerator
from utils import _get_default_candidate_generator_generator

from config import *

import os

draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    use_flash_attention_2=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    use_flash_attention_2=True,
    device_map="auto",
)

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

ds = load_dataset(dataset_name, split=dataset_split)

def save_file(stats):
    if not(os.path.exists(output_file)):
        directory = os.path.dirname(output_file)
        os.makedirs(directory, exist_ok=True)
    stats_file = open(output_file, "w+")
    stats_file.write(json.dumps(stats))
    stats_file.close()

def print_update(dictionary):
    for key in dictionary:
        if not (key.endswith("text")):
            print(f"{key}: {dictionary[key]}")
    print("======")

stats = []
regular_get_candidate_generator = model._get_candidate_generator

for row_idx, row in tqdm(enumerate(ds)):
    output = {}
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


    code_tokens = tokenizer(row["code"], return_tensors="pt").to(model.device)
    starting_input_tokens = inputs.shape[-1]
    max_new_tokens = code_tokens.input_ids.shape[-1] + 512

    # Assisted decoding
    if USE_ASSISTED_DECODING:
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
            temperature=temperature
        )
        end_time = time.perf_counter()
        output["assisted_decoding"] = end_time - start_time

    # Regular decoding
    if USE_REGULAR_DECODING:
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
            temperature=temperature
        )
        end_time = time.perf_counter()
        output["regular_decoding"] = end_time - start_time

    for lt in lookup_tokens:
        # Using HuggingFace Prompt Lookup Decoding
        if USE_PROMPT_LOOKUP_DECODING:
            model._get_candidate_generator = (regular_get_candidate_generator).__get__(
                model, type(model)
            )
            start_time = time.perf_counter()
            pld_output = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                max_matching_ngram_size=max_matching_ngram_size,
                return_dict_in_generate=True,
                output_scores=True,
                prompt_lookup_num_tokens=lt,
                use_cache=True,
                temperature=temperature,
            )
            end_time = time.perf_counter()

            output[f"pld_{lt}"] = end_time - start_time
            output[f"pld_{lt}_text"] = tokenizer.decode(
                pld_output.sequences[0, starting_input_tokens:]
            )

        for mdt in model_draft_tokens:
            # Using Two Layer Lookup Decoding
            if USE_TWO_LAYER_LOOKUP_DECODING:
                two_layer_candidate_generator = TwoLayerLookupCandidateGenerator(
                    tokenizer,
                    inputs.shape[-1],
                    draft_model,
                    inputs,
                    code_tokens.input_ids.tolist()[0],
                    ngram_size=5,
                    num_pred_tokens=lt,
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
                    temperature=temperature,
                    return_dict_in_generate=True,
                )
                end_time = time.perf_counter()
                output[f"method_{lt}_{mdt}"] = end_time - start_time
                output[f"method_{lt}_{mdt}_text"] = tokenizer.decode(
                    two_layer_out.sequences[0, starting_input_tokens:]
                )

    stats.append(output)
    print_update(output)
    save_file(stats)

save_file(stats)
