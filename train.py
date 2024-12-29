# !pip install datasets  # uncomment in colab

import os
from datasets import load_dataset, Dataset
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

debug = True


def main():
    # ---------------------------------------
    # 1) Load your SentencePiece model as a Tokenizer object in memory
    # ---------------------------------------
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./ConvertedTokenizer")

    # ---------------------------------------
    # 2) Load your dataset
    # ---------------------------------------
    if debug:
        # Enable streaming FOR DEBUGGING
        dataset_stream = load_dataset(
            "wikipedia",
            "20220301.en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        small_stream = dataset_stream.take(
            1000
        )  # take first 1000 samples for debugging
        # To "materialize" them in memory as a small list or dataset:
        samples = list(small_stream)  # "materialize" 1000 items
        dataset = Dataset.from_list(samples)
    else:
        dataset = load_dataset(
            "wikipedia", "20220301.en", split="train[:1%]", trust_remote_code=True
        )

    # ---------------------------------------
    # 3. Create a ~14M GPT-NeoX config
    # ---------------------------------------
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

    # ---------------------------------------
    # 5. Tokenize the dataset
    # ---------------------------------------
    # Wikipedia data typically has a "text" field.
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=1024,  # or smaller to save memory
        )

    # We remove the original "text" column after tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,  # adjust if you have more or fewer CPU cores
        remove_columns=["text"],
    )

    # ---------------------------------------
    # 6. (Optional) Group texts into 1024-token chunks
    # ---------------------------------------
    # This can improve throughput by reducing the overhead from short sequences.
    block_size = 1024

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

    tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

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
        fp16=True,  # set True if your GPU supports FP16
        gradient_accumulation_steps=4,
        report_to="none",  # or "tensorboard", "wandb", etc.
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=(
            tokenized_dataset["validation"]
            if "validation" in tokenized_dataset
            else None
        ),
    )

    # ---------------------------------------
    # 8. Train the model
    # ---------------------------------------
    trainer.train()


if __name__ == "__main__":
    main()
