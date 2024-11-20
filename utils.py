from transformers.generation.candidate_generator import (
    CandidateGenerator,
)

def _get_default_candidate_generator_generator(generator: CandidateGenerator):
    def _get_candidate_generator(self, **kwargs):
        return generator

    return _get_candidate_generator