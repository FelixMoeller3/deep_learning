################### COLAB ###################
# !pip install datasets
# !pip install mwparserfromhell
# !pip install evaluate
# from google.colab import drive
# drive.mount('/content/drive')
################### COLAB ###################

import os
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import evaluate
import numpy as np
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
    hidden_size=256,  # smaller hidden dim
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

    # 1) Convert logits to predictions
    predictions = top10_ids[..., 0]  # shape: (batch_size, seq_len)

    # 2) Flatten predictions & labels for normal accuracy / F1
    flattened_preds = predictions.flatten()
    flattened_labels = labels.flatten()

    # Basic Metrics: Accuracy & F1
    accuracy = accuracy_score(flattened_labels, flattened_preds)
    f1 = f1_score(flattened_labels, flattened_preds, average="weighted")

    # 3) Compute Top-5 and Top-10 accuracy
    #    a) Sort along the vocab dimension; b) extract the last 5 or 10
    top5_ids = top10_ids[..., :5]  # shape: (batch_size, seq_len, 5)

    # We'll iterate only over the same valid positions we used above
    total_valid = len(flattened_labels)
    top5_correct = 0
    top10_correct = 0

    # We need to match the batch/seq positions back to the flattened valid indices
    # One approach is to reshape again and step carefully.
    # Another approach is to do everything in a 2D loop.
    # Here, let's do a 2D loop for clarity:
    b_sz, seq_len = predictions.shape
    idx = 0
    for b in range(b_sz):
        for s in range(seq_len):
            if labels[b, s] == -100:  # ignore
                continue
            label_id = labels[b, s]
            # check top-5
            if label_id in top5_ids[b, s]:
                top5_correct += 1
            # check top-10
            if label_id in top10_ids[b, s]:
                top10_correct += 1
            idx += 1  # index for valid positions

    top5_acc = top5_correct / total_valid if total_valid > 0 else 0.0
    top10_acc = top10_correct / total_valid if total_valid > 0 else 0.0

    # 4) Type-Token Ratio (TTR) and Average Subword Length
    #    We compute these from the predicted tokens.
    unique_pred_ids = np.unique(flattened_preds)
    ttr = (
        len(unique_pred_ids) / float(len(flattened_preds))
        if len(flattened_preds) > 0
        else 0.0
    )

    # For average subword length, decode each predicted token (careful with large sets!)
    subword_lengths = []
    for pred_id in flattened_preds:
        # Convert ID back to the subword string
        token_str = tokenizer.decode([pred_id], skip_special_tokens=True)
        subword_lengths.append(len(token_str))
    avg_subword_length = np.mean(subword_lengths) if subword_lengths else 0.0

    return {
        "accuracy": accuracy,
        "f1": f1,
        "top5_accuracy": top5_acc,
        "top10_accuracy": top10_acc,
        "type_token_ratio": ttr,
        "avg_subword_length": avg_subword_length,
    }


model.eval()
torch.no_grad()

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
eval_results["eval_perplexity"] = np.exp(eval_results["eval_loss"])
print(eval_results)
# 11) (Optional) Save the final model + tokenizer explicitly
