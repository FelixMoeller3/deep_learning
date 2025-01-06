from datasets import load_dataset, Dataset, DatasetDict
import sys

if len(sys.argv) != 2:
    print("Usage: python count_tokens.py <language>")
    sys.exit(1)

language = sys.argv[1]


def count_tokens(dataset_dict):
    token_counts = {}
    for split in ["train", "test"]:
        # count tokens in split
        token_counts[split] = sum(
            len(token) for token in dataset_dict[split]["input_ids"]
        )
    return token_counts


# Load your dataset
dataset_dict = DatasetDict.load_from_disk(
    f"/Users/hanno/Documents/DL/deep_learning/datasets/{language}_lead_ready_to_train"
)

# Count tokens in train and test splits
token_counts = count_tokens(dataset_dict)
print(token_counts)
