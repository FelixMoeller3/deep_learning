################### COLAB ###################
# !pip install datasets
# !pip install mwparserfromhell
# !pip install evaluate
# from google.colab import drive
# drive.mount('/content/drive')
################### COLAB ###################

from datasets import Dataset, DatasetDict
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import numpy as np
import pprint
from sklearn.metrics import accuracy_score, f1_score

torch.cuda.empty_cache()

# so I have this working fine, but gets stuck in the train-test split
# test, save model too. english, suffixing. RUNNING HERE
# let it start training and then see, if so then adjust train size, also eval strategy and add the real tokenizer and so on

debug = False
PREFIX = "/content/drive/Shareddrives/DeepLearning/"

SEED = "9336"
TOKENIZER = "lead"
LANGUAGE = "fi"
MODEL_SIZE = "14m"  # 14m, 60m
TOKENIZER_PATH = f"{PREFIX}Tokenizers/{LANGUAGE}/{LANGUAGE}_{TOKENIZER}_tokenizer"  # CHECK for correct tokenizer
DATASET_PATH = f"{PREFIX}Datasets/{LANGUAGE}/{LANGUAGE}_{TOKENIZER}_ready_to_train"  # CHECK for correct dataset
SEQUENCE_LENGTH = 1024

hidden_size_mapping = {
    "14m": 256,
    "60m": 512,
}


# 1) Load your SentencePiece (or other) tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
# 2) Load a small portion of Wikipedia for debugging
dataset = DatasetDict.load_from_disk(DATASET_PATH)  # CHECK whether to use DatasetDict
if debug:
    small_stream = dataset["train"].select(range(int(100)))
    samples = list(small_stream)
    dataset = Dataset.from_list(samples)

# 3) Create a ~14M param GPT-NeoX config
config = GPTNeoXConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=hidden_size_mapping[MODEL_SIZE],  # smaller hidden dim
    num_hidden_layers=6,  # fewer layers
    num_attention_heads=8,
    intermediate_size=1024,
    max_position_embeddings=2048,
)


# 4) Build a GPT-NeoX model (similar to a Pythia-style model)
model = GPTNeoXForCausalLM.from_pretrained(
    f"{PREFIX}Models/{LANGUAGE}/model_{MODEL_SIZE}_{TOKENIZER}_{SEED}"
)
# model = GPTNeoXForCausalLM(config)

# 7) Use a DataCollator that sets 'labels' for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # for causal LM
)

# 8) Define training arguments
training_args = TrainingArguments(
    output_dir=f"{PREFIX}Models/{LANGUAGE}/model_{MODEL_SIZE}_{TOKENIZER}_{SEED}_checkpoints",  # CHECK for correct path
    overwrite_output_dir=True,
    num_train_epochs=1,  # ADJUST EPOCHS HERE
    per_device_train_batch_size=4,
    evaluation_strategy="no",
    eval_steps=1000,
    save_steps=1000,
    logging_steps=100,  # change to 200 or so
    learning_rate=2e-4,
    warmup_steps=500,
    weight_decay=0.01,
    bf16=False,
    fp16=True,  # CHECK true for GPU
    gradient_accumulation_steps=4,
    report_to="tensorboard",
)


def preprocess_logits_for_metrics(logits, labels):
    # logits shape: (batch, seq_len, vocab_size)
    # gather top-10 indices
    top10_ids = torch.topk(logits, k=10, dim=-1).indices  # (batch, seq_len, 10)
    return top10_ids


def compute_metrics(eval_pred):
    """
    Calculate accuracy, weighted F1, top-5 accuracy, top-10 accuracy,
    type-token ratio (TTR), and average subword length from predictions.
    """
    top10_ids, labels = eval_pred  # shape: (batch_size, seq_len, vocab_size)

    top10_ids = top10_ids.reshape(-1,10)
    flattened_labels = labels.flatten()
    # 1) Convert logits to predictions
    predictions = top10_ids[..., 0] # shape: (batch_size*seq_len,)

    # Basic Metrics: Accuracy & F1
    accuracy = accuracy_score(flattened_labels, predictions)
    f1 = f1_score(flattened_labels, predictions, average="weighted")

    # 3) Compute Top-5 and Top-10 accuracy
    top_5_accuracy = np.equal(top10_ids[:,:5],flattened_labels[:,None]).any(axis=1).mean()
    top_10_accuracy = np.equal(top10_ids,flattened_labels[:,None]).any(axis=1).mean()

    # 4) Type-Token Ratio (TTR) and Average Subword Length
    #    We compute these from the tokens in the original dataset.
    ttr = 0
    for sequence in labels:
        ttr+= len(np.unique(sequence))
    ttr/= SEQUENCE_LENGTH*len(labels)

    # For average subword length, decode each predicted token (careful with large sets!)
    tokens_str = tokenizer.decode([*range(tokenizer.vocab_size)], skip_special_tokens=True)
    avg_subword_length = sum(map(lambda x: len(x),tokens_str)) / tokenizer.vocab_size

    return {
        "accuracy": accuracy,
        "f1": f1,
        "top5_accuracy": top_5_accuracy,
        "top10_accuracy": top_10_accuracy,
        "type_token_ratio(tokenizer and dataset)": ttr,
        "avg_token_length(only tokenizer)": avg_subword_length,
    }

# 9) Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],  # CHECK if using DatasetDict, select ["train"]
    eval_dataset=dataset["test"],  # CHECK consider using the test set of the dataset
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# 10) Train
eval_results = trainer.evaluate()
eval_results['eval_perplexity'] = np.exp(eval_results['eval_loss'])
pprint.pp(eval_results)
# 11) (Optional) Save the final model + tokenizer explicitly
