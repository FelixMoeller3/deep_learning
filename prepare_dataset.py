# so I have this working fine, but gets stuck in the train-test split
# test, save model too. english, suffixing. RUNNING HERE
# let it start training and then see, if so then adjust train size, also eval strategy and add the real tokenizer and so on
import os
from datasets import load_dataset, Dataset
import numpy as np
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


########## CHANGE THESE ##########
LANGUAGE = "fi"  # fi, sw, zh, en
TOKENIZER = "lead"  # lead, trail
##################################


debug = False
PREFIX = "./"
TOKENIZER_PATH = PREFIX + f"tokenizers/{LANGUAGE}_{TOKENIZER}_tokenizer"
DATASET_PATH = PREFIX + f"datasets/{LANGUAGE}"

# 1) Load your SentencePiece (or other) tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)

# 2) Load a small portion of Wikipedia for debugging
dataset = load_dataset(
    "wikipedia",
    language=LANGUAGE,
    date="20250101",
    split="train",
    trust_remote_code=True,
)
# dataset = Dataset.load_from_disk(DATASET_PATH)
dataset = dataset.shuffle(seed=42)
if debug:
    small_stream = dataset.select(range(int(1000)))
    samples = list(small_stream)
    dataset = Dataset.from_list(samples)

# 3) Create a ~14M param GPT-NeoX config
config = GPTNeoXConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,  # smaller hidden dim
    num_hidden_layers=6,  # fewer layers
    num_attention_heads=8,
    intermediate_size=1024,
    max_position_embeddings=2048,
)

# 4) Build a GPT-NeoX model (similar to a Pythia-style model)
model = GPTNeoXForCausalLM(config)


def preprocess_function(examples):
    # split on \n\nSee also\n for wikipedia and remove the last part
    examples["text"] = [
        article.split("\n\nSee also\n")[0] for article in examples["text"]
    ]
    return examples


dataset = dataset.map(preprocess_function, batched=True)


# 5) Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"])


# Remove "text" column after tokenizing
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["text"],
)

print("Tokenized dataset example:", tokenized_dataset[0])

# 6) Flatten all tokens into one big list and chunk them into 1024-token blocks
all_ids = np.concatenate([example["input_ids"] for example in tokenized_dataset])

block_size = 1024
total_length = (len(all_ids) // block_size) * block_size
all_ids = all_ids[:total_length]  # drop leftover tokens
chunks = np.split(all_ids, total_length // block_size)

# Build a new Dataset from these 1024-token chunks
chunked_dataset = Dataset.from_dict(
    {
        "input_ids": chunks,
        "attention_mask": np.ones((len(chunks), block_size), dtype=np.int32).tolist(),
    }
)

print("Number of 1024-token chunks:", len(chunked_dataset))

# split train and test
chunked_dataset = chunked_dataset.train_test_split(test_size=0.1)

# save the chunked dataset
chunked_dataset.save_to_disk(DATASET_PATH + f"_{TOKENIZER}_ready_to_train")
