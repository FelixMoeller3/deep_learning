import os
from datasets import load_dataset, Dataset
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score,accuracy_score




full_dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True,  # Enable streaming mode
        trust_remote_code=True,
    )

small_dataset = []
for i, sample in enumerate(full_dataset):
    small_dataset.append(sample)
    if i >= 100:
        break
dataset = Dataset.from_dict(
    {
        key: [example[key] for example in small_dataset]
        for key in small_dataset[0]
    }
)

tokenizer = PreTrainedTokenizerFast.from_pretrained("./ConvertedTokenizer")

config = GPTNeoXConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=1024,
    max_position_embeddings=2048,
)

# ---------------------------------------
# 4. Build a GPT-NeoX model from scratch
# ---------------------------------------
model = GPTNeoXForCausalLM(config)
#model.forward()

def preprocess_function(examples):
    # split on \n\nSee also\n for wikipedia and remove the last part
    examples["text"] = [
        article.split("\n\nSee also\n")[0] for article in examples["text"]
    ]
    return examples

dataset = dataset.map(preprocess_function, batched=True)

block_size = 1024

def chunk_examples(batch, block_size=1024):
    chunks = []
    titles = []
    ids = []
    for sentences in batch["text"]:
        chunk = [
            sentences[i : i + block_size]
            for i in range(0, len(sentences), block_size)
        ]
        chunks += chunk
        titles += [batch["title"]] * len(chunk)
        ids += [batch["id"]] * len(chunk)
    return {"text": chunks, "title": titles, "id": ids}

chunk_dataset = dataset.map(
    chunk_examples, batched=True, remove_columns=["text", "title", "url", "id"]
)

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = chunk_dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=["text", "title", "id"],
)

# ---------------------------------------
# 6. (Optional) Group texts into 1024-token chunks
# ---------------------------------------
# This can improve throughput by reducing the overhead from short sequences.

def group_texts(examples):
    concatenated_ids = []
    for ids in examples["input_ids"]:
        concatenated_ids.extend(ids)

    # Drop the remainder so everything is a full block
    total_length = (len(concatenated_ids) // block_size) * block_size
    result = []
    for i in range(0, total_length, block_size):
        result.append(concatenated_ids[i : i + block_size])

    return {"input_ids": result, "attention_mask": [[1] * block_size] * len(result)}

# tokenized_dataset = tokenized_dataset.map(
#     group_texts,
#     batched=True,
#     batch_size=1000,
#     num_proc=4,
# )

# ---------------------------------------
# 7. Create training arguments
# ---------------------------------------
training_args = TrainingArguments(
    output_dir="path/to/checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=3,  # adjust as desired
    per_device_train_batch_size=4,  # depends on your GPU memory
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_steps=200,
    learning_rate=2e-4,
    warmup_steps=500,
    weight_decay=0.01,
    bf16=False,  # set True if your GPU supports BF16
    fp16=False,  # set True if your GPU supports FP16
    gradient_accumulation_steps=4,
    report_to="none",  # or "tensorboard", "wandb", etc.
)

# split the dataset into training and validation
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# ---------------------------------------
# 8. Train the model
# ---------------------------------------
#trainer.train()
trainer.evaluate()
print(tokenized_dataset["test"][0].keys())