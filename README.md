# Two-layer Speculative Editing

This repository contains the implementation of a two-layer speculative editing system. The system is designed to generate and evaluate code edits using a combination of different decoding strategies.

## Repository Structure

- `config.py`: This file contains the `Config` class and functions to load configurations.
- `configs/`: Directory containing JSON configuration files.
- `evaluate.py`: Main script to run the evaluation.
- `evaluation/`: Directory containing evaluation results.
- `two_layer_candidate_generator.py`: Contains the `TwoLayerLookupCandidateGenerator` class.
- `utils.py`: Utility functions for the project.
- `visualize.py`: Script for visualizing the results.
- `README.md`: Project documentation.

## Useful Commands

### Running an Evaluation

To run an evaluation, use the following command:

```sh
python evaluate.py path/to/config/file.json
```

Sample configurations can be found in the `configs` folder.

### Visualizing Results

To visualize a certain set of runs, use the following command:

```sh
python visualize.py path/to/output/file.json output/file.png
```

This will save a visualization to the specified folder.

## Output File Formatting

The dataset is loaded directly from HuggingFace without shuffling. Since it's run sequentially, you can expect the order of the outputs to line up with the order of the inputs. The generated list is a list with the attributes being as follows:

```json
{
    "method_PLDTokens_NumDraftModelRuns": <time taken>,
    "method_PLDTokens_NumDraftModelRuns_text": <output text>,
    "pld_PLDTokens": <time taken>,
    "pld_text": <output text>,
    "assisted_decoding": <time taken>,
    "regular_decoding": <time taken>
}
```

## Configuration File Details

Each configuration file in the `configs` directory contains the following attributes:

- `seed`: Random seed for reproducibility.
- `temperature`: Temperature setting for the model.
- `draft_model_name`: Name of the draft model.
- `model_name`: Name of the main model.
- `dataset_name`: Name of the dataset.
- `dataset_split`: Split of the dataset (e.g., train, test).
- `output_file`: Path to the output file where results will be saved.
- `model_draft_tokens`: List of draft tokens for the model.
- `lookup_tokens`: List of lookup tokens for the model.
- `max_matching_ngram_size`: Maximum size of matching n-grams.
- `USE_ASSISTED_DECODING`: Boolean flag to use assisted decoding.
- `USE_REGULAR_DECODING`: Boolean flag to use regular decoding.
- `USE_PROMPT_LOOKUP_DECODING`: Boolean flag to use prompt lookup decoding.
- `USE_TWO_LAYER_LOOKUP_DECODING`: Boolean flag to use two-layer lookup decoding.

## Example Configuration

Here is an example configuration from `configs/qwen32b_5k.json`:

```json
{
    "seed": 42,
    "temperature": 0.0,
    "draft_model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "dataset_name": "vdaita/edit_time_5k",
    "dataset_split": "train",
    "output_file": "evaluation/results/qwen_test_5k.json",
    "model_draft_tokens": [
        1,
        2,
        4,
        8,
        12
    ],
    "lookup_tokens": [
        120
    ],
    "max_matching_ngram_size": 10,
    "USE_ASSISTED_DECODING": false,
    "USE_REGULAR_DECODING": false,
    "USE_PROMPT_LOOKUP_DECODING": true,
    "USE_TWO_LAYER_LOOKUP_DECODING": true
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.