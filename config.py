seed = 42
temperature = 0.0

draft_model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
model_name = "deepseek-ai/deepseek-coder-33b-instruct"

dataset_name = "vdaita/edit_time_1k"
dataset_split = "train"

output_file = "evaluation/results/original_code_deepseek_test_1k_with_regular.json"
model_draft_tokens = [1]
lookup_tokens = [20]

max_matching_ngram_size = 5

USE_ASSISTED_DECODING = True
USE_REGULAR_DECODING = True
USE_PROMPT_LOOKUP_DECODING = True
USE_TWO_LAYER_LOOKUP_DECODING = True