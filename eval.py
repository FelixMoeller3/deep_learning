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

# so I have this working fine, but gets stuck in the train-test split
# test, save model too. english, suffixing. RUNNING HERE
# let it start training and then see, if so then adjust train size, also eval strategy and add the real tokenizer and so on

debug = False
PREFIX = "/content/drive/Shareddrives/DeepLearning/"

SEED = '9336'
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
model = GPTNeoXForCausalLM.from_pretrained(f"{PREFIX}Models/{LANGUAGE}/model_{MODEL_SIZE}_{TOKENIZER}_{SEED}")
#model = GPTNeoXForCausalLM(config)

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
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def compute_metrics(eval_pred):
    "Calculate metrics and accuracy"
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    # Berechnung der Metriken
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "f1": f1,
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
print(eval_results)
# 11) (Optional) Save the final model + tokenizer explicitly

