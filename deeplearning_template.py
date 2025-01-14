###### DO NOT USE THIS TEMPLATE FOR TRAINING ######
from datasets import load_dataset, Dataset
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

debug = True

# 1) Load your SentencePiece (or other) tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("/content/drive/MyDrive/ConvertedTokenizer")

# 2) Load a small portion of Wikipedia for debugging
if debug:
    dataset_stream = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    # Take 1000 samples for debug
    small_stream = dataset_stream.take(1000)
    samples = list(small_stream)
    dataset = Dataset.from_list(samples)
else:
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]", trust_remote_code=True)

# 3) Create a ~14M param GPT-NeoX config (CHANGE FOR EACH MODEL SIZE, LOOK UP/CHATGPT PARAMS)
config = GPTNeoXConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,         # smaller hidden dim
    num_hidden_layers=6,     # fewer layers
    num_attention_heads=8,
    intermediate_size=1024,
    max_position_embeddings=2048,
)

# 4) Build a GPT-NeoX model (similar to a Pythia-style model)
model = GPTNeoXForCausalLM(config)

# 5) Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=1024)

# Remove "text" column after tokenizing
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["text"],
)

print("Tokenized dataset example:", tokenized_dataset[0])

# 6) Flatten all tokens into one big list and chunk them into 1024-token blocks
all_ids = []
for example in tokenized_dataset:
    all_ids.extend(example["input_ids"])

block_size = 1024
total_length = (len(all_ids) // block_size) * block_size
all_ids = all_ids[:total_length]  # drop leftover tokens
chunks = [
    all_ids[i : i + block_size]
    for i in range(0, total_length, block_size)
]

# Build a new Dataset from these 1024-token chunks
chunked_dataset = Dataset.from_dict({
    "input_ids": chunks,
    "attention_mask": [[1]*block_size for _ in range(len(chunks))]
})

print("Number of 1024-token chunks:", len(chunked_dataset))

# 7) Use a DataCollator that sets 'labels' for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # for causal LM
)

# 8) Define training arguments
training_args = TrainingArguments(
    output_dir="pythia-14m-checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=3,              # Increase if you want more training
    per_device_train_batch_size=4,   # Adjust according to GPU memory
    evaluation_strategy="no",        # No eval for now (to avoid the error)
    eval_steps=1000,
    save_steps=1000,
    logging_steps=200,
    learning_rate=2e-4,
    warmup_steps=500,
    weight_decay=0.01,
    bf16=False,   # Set True if GPU supports BF16
    fp16=True,    # Set True if GPU supports FP16
    gradient_accumulation_steps=4,
    report_to="none",  # or "tensorboard", "wandb", etc.
)

# 9) Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=chunked_dataset,
    eval_dataset=None,  # No eval dataset here
    data_collator=data_collator,
)

# 10) Train
trainer.train()

#check the train.py
