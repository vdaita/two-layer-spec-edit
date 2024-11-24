# SPDX-FileCopyrightText: (c) iSE UIUC Research Group
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass


@dataclass
class Config:
    seed: int
    temperature: float
    draft_model_name: str
    model_name: str
    dataset_name: str
    dataset_split: str
    output_file: str
    model_draft_tokens: list
    lookup_tokens: list
    max_matching_ngram_size: int
    USE_ASSISTED_DECODING: bool
    USE_REGULAR_DECODING: bool
    USE_PROMPT_LOOKUP_DECODING: bool
    USE_TWO_LAYER_LOOKUP_DECODING: bool


def load_config_from_json(json_file: str) -> Config:
    with open(json_file, "r") as file:
        data = json.load(file)
    return Config(**data)


# Example usage
# config = load_config_from_json('config.json')
# print(config)

"""
Example configuration JSON file.
{
    "seed": 42,
    "temperature": 0.7,
    "draft_model_name": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "model_name": "deepseek-ai/deepseek-coder-33b-instruct",
    "dataset_name": "vdaita/edit_time_1k",
    "dataset_split": "train",
    "output_file": "evaluation/results/original_code_deepseek_test_1k_with_regular.json",
    "model_draft_tokens": [1],
    "lookup_tokens": [20],
    "max_matching_ngram_size": 5,
    "USE_ASSISTED_DECODING": true,
    "USE_REGULAR_DECODING": true,
    "USE_PROMPT_LOOKUP_DECODING": true,
    "USE_TWO_LAYER_LOOKUP_DECODING": true
}
"""
