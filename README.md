# Two-layer speculative editing

Repository structure:
`config.py`: This file contains the `Config` class and functions to load configurations.
`configs/`: Directory containing JSON configuration files.
`evaluate.py`: Main script to run the evaluation.
`two_layer_candidate_generator.py`: Contains the `TwoLayerLookupCandidateGenerator` class.
`utils.py`: Utility functions for the project.
`visualize.py`: Script for visualizing the results.
`README.md`: Project documentation.

# Useful commands
To run an evaluation run: ```python evaluate.py path/to/config/file.json```. Sample configurations can be found in the configs folder.

To visualize a certain set of runs, run ``` python visualize.py path/to/output/file.json output/file.png``` to save a visualization to a particular folder. 

# Output file formatting
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

