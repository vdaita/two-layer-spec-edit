import os
import json

from transformers.generation.candidate_generator import (
    CandidateGenerator,
)

def _get_default_candidate_generator_generator(generator: CandidateGenerator):
    def _get_candidate_generator(self, **kwargs):
        return generator

    return _get_candidate_generator

def save_file(stats, output_file):
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