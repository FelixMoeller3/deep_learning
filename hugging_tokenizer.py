from tokenizers import SentencePieceBPETokenizer
from datasets import load_dataset
import os


def train_bpe(model_prefix: str, whitespace: bool):
    # Load a subset of the Wikipedia dataset for training
    dataset = load_dataset(
        "wikipedia", "20220301.simple", split="train"
    )  # Using 1% for example

    # Extract text data from the dataset
    texts = dataset["text"]
    tokenizer = SentencePieceBPETokenizer(add_prefix_space=whitespace)
    # Add special tokens
    special_tokens = ["<unk>", "<s>", "</s>"]
    # Train tokenizer from the dataset
    tokenizer.train_from_iterator(
        texts,
        vocab_size=16000,
        show_progress=True,
        special_tokens=special_tokens,
    )

    # Save tokenizer to a file
    tokenizer.save(f"{model_prefix}.json")
    print(f"Tokenizer saved as {model_prefix}.json")


def test_tokenizer(tokenizer_path):
    # Load the tokenizer
    tokenizer = SentencePieceBPETokenizer(tokenizer_path)

    # Test the tokenizer
    text = "Hello, my dog is cute"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)

    print(f"Original text: {text}")
    print(f"Encoded IDs: {encoded.ids}")
    print(f"Decoded text: {decoded}")


def main():
    model_prefix = "sentencepiece_en_prefix"
    whitespace = True

    # Train and save the tokenizer
    train_bpe(model_prefix, whitespace)

    # Test the tokenizer
    tokenizer_path = f"{model_prefix}.json"
    test_tokenizer(tokenizer_path)


if __name__ == "__main__":
    main()
