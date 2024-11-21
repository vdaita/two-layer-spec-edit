from fire import Fire
import numpy as np
import matplotlib.pyplot as plt
import json

def visualize(data_path: str, output_path: str = "output.png", title: str = None):
    print(f"Visualizing {data_path} to {output_path}")
    if not(title):
        title = f"Results visualizations for {data_path}"

    results = json.loads(open(data_path).read())
    means = {}
    std = {}
    for test_case in results:
        baseline = test_case["pld_20"]
        for eval_type in test_case:
            if eval_type.endswith("text"):
                continue
            if eval_type not in means:
                means[eval_type] = []
            means[eval_type].append(baseline / test_case[eval_type])
            
    for eval_type in means:
        std[eval_type] = np.std(means[eval_type])
        means[eval_type] = np.mean(means[eval_type])

    line_groups = {}
    for eval_type in means:
        if eval_type.startswith("pld"):
            lookup_amount = int(eval_type.split("_")[1])
            if not lookup_amount in line_groups:
                line_groups[lookup_amount] = {}
            line_groups[lookup_amount][0] = (means[eval_type], std[eval_type])
        elif eval_type.startswith("method"):
            lookup_amount = int(eval_type.split("_")[1])
            draft_tokens = int(eval_type.split("_")[2])
            if not lookup_amount in line_groups:
                line_groups[lookup_amount] = {}
            line_groups[lookup_amount][draft_tokens] = (
                means[eval_type],
                std[eval_type],
            )

    plt.figure(figsize=(10, 6))
    # for eval_type in means:
    #     if not(eval_type.startswith("pld")) and not(eval_type.startswith("method")):
    #         # just draw a straight line
    #         plt.axhline(y=means[eval_type], label=eval_type, linestyle="--")

    for lookup_amount in line_groups:
        x = sorted(line_groups[lookup_amount].keys())
        y = [line_groups[lookup_amount][draft_token][0] for draft_token in x]
        # yerr = [line_groups[lookup_amount][draft_token][1] for draft_token in x]
        plt.plot(x, y, label=f"Lookups: {lookup_amount}")
        # plt.fill_between(
        #     x, np.array(y) - np.array(yerr), np.array(y) + np.array(yerr), alpha=0.2
        # )

    plt.title(title)
    plt.xlabel("Draft Tokens")
    plt.ylabel("Mean Ratio")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Fire(visualize)