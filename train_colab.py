##############  COLAB  #####################
# !pip install mwparserfromhell
# !pip install datasets
# !pip install wandb
# import wandb
# import os
# os.environ["WANDB_API_KEY"] = REDACTED
# from google.colab import drive
# drive.mount('/content/drive')
############################################

###############  SEED  #####################
import random
import numpy as np
import torch
from transformers import set_seed

SEED = 356
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(SEED)
############################################


# so I have this working fine, but gets stuck in the train-test split
# test, save model too. english, suffixing. RUNNING HERE
# let it start training and then see, if so then adjust train size, also eval strategy and add the real tokenizer and so on
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

debug = False
PREFIX = "/content/drive/Shareddrives/DeepLearning/"

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
model = GPTNeoXForCausalLM(config)

# 7) Use a DataCollator that sets 'labels' for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # for causal LM
)

# ------ Initialize W&B project (optional, but recommended) ------
wandb.init(
    project="deeplearning",  # change to your W&B project name
    name=f"model_{LANGUAGE}_{MODEL_SIZE}_{TOKENIZER}_{SEED}"   # name for your specific run
)
# 8) Define training arguments
training_args = TrainingArguments(
    output_dir=f"{PREFIX}Models/{LANGUAGE}/model_{MODEL_SIZE}_{TOKENIZER}_{SEED}_checkpoints",  # CHECK for correct path
    overwrite_output_dir=True,
    num_train_epochs=0.42,  # ADJUST EPOCHS HERE
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
    report_to=["wandb", "tensorboard"],
    run_name=f"model_{LANGUAGE}_{MODEL_SIZE}_{TOKENIZER}_{SEED}"
)

# 9) Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],  # CHECK if using DatasetDict, select ["train"]
    eval_dataset=None,  # CHECK consider using the test set of the dataset
    data_collator=data_collator,
)

# 10) Train
trainer.train()

# 11) (Optional) Save the final model + tokenizer explicitly
trainer.save_model(
    f"{PREFIX}Models/{LANGUAGE}/model_{MODEL_SIZE}_{TOKENIZER}_{SEED}"
)  # CHECK for correct path
wandb.finish()
