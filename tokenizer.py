from datasets import load_dataset
from transformers import convert_slow_tokenizer, PreTrainedTokenizerFast
import tokenizers
import sentencepiece as spm
import pandas as pd
import os
import tempfile
import json


def prepare_dataset_for_training(dataset):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as temp_file:
        # Write each text entry to the file
        for text in dataset["text"]:
            temp_file.write(text + "\n")  # Each entry on a new line
        temp_file_path = temp_file.name
    return temp_file_path


def subsample_dataset(dataset):
    # Shuffle and select 10% of the data
    return dataset.shuffle(seed=42).select(range(int(0.1 * len(dataset))))


def train_sentencepiece(model_prefix: str, whitespace: bool):
    # Load the dataset
    dataset = load_dataset("wikipedia", "20220301.simple", split="train")

    # Subsample the dataset
    dataset = subsample_dataset(dataset)

    # Convert to Pandas DataFrame for processing (if needed)
    df = dataset.to_pandas()

    # Prepare the dataset for training
    dataset_file_path = prepare_dataset_for_training(df)

    # Define a prefix for the model files

    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=dataset_file_path,
        vocab_size=16000,
        model_type="bpe",
        treat_whitespace_as_suffix=whitespace,
        add_dummy_prefix=False,
        model_prefix=model_prefix,  # Set the model prefix
    )
    # Load the trained model
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    # Extract vocabulary
    vocab_size = sp.get_piece_size()  # Number of pieces in the model
    vocab = {i: sp.id_to_piece(i) for i in range(vocab_size)}

    # Prepare a dictionary to save the model and vocabulary
    model_data = {"model_type": "bpe", "vocab_size": vocab_size, "vocab": vocab}

    # Prepare a dictionary to save the model and vocabulary
    model_data = {"model_type": "bpe", "vocab_size": len(vocab), "vocab": vocab}

    # Save the model data as JSON
    with open(f"{model_prefix}.json", "w") as json_file:
        json.dump(model_data, json_file, ensure_ascii=False, indent=4)

    # Clean up the temporary file
    os.remove(dataset_file_path)


def tokenizer_to_huggingface(model_prefix):
    # Load the SentencePiece model
    spm_tokenizer = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    spm_tokenizer.vocab_file = f"{model_prefix}.model"

    # Convert SentencePiece tokenizer to Hugging Face format
    spm_converter = convert_slow_tokenizer.SpmConverter(spm_tokenizer)
    converted = spm_converter.converted()

    # Create a directory to save the tokenizer files
    output_dir = "ConvertedTokenizer"
    os.makedirs(output_dir, exist_ok=True)

    # Save the converted tokenizer to JSON
    converted.save(os.path.join(output_dir, "tokenizer.json"))

    # Save the tokenizer configuration
    tokenizer_config = {
        "model_max_length": 1024,
        "clean_up_tokenization_spaces": False,
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "mask_token": "<MASK>",
        "padding_side": "right",
        "truncation_side": "left",
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=4)

    # Load the PreTrainedTokenizerFast from the directory
    tok = PreTrainedTokenizerFast.from_pretrained(output_dir)

    # set pre tokenizer to
    # "Metaspace", replacement "▁", prepend_scheme "never", split false
    tok.backend_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="never", split=False
    )
    # tok.backend_tokenizer.normalizer =

    # set decoder to
    # "Metaspace", replacement "▁", prepend_scheme "never", split false

    tok.save_pretrained(output_dir)


def test_tokenizer(model_prefix, test_text):
    # Load the trained SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")

    # Encode the test text
    encoded = sp.EncodeAsPieces(test_text)
    token_ids = sp.EncodeAsIds(test_text)
    print(token_ids)
    # Decode the encoded text
    decoded = sp.DecodePieces(encoded)

    return encoded, decoded


def test_tokenizer_hf(model_prefix, test_text):
    # Load the Hugging Face tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./ConvertedTokenizer")
    # Tokenize the test text
    tokenized = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text, add_special_tokens=True)
    print(token_ids)
    # Decode the token IDs
    decoded = tokenizer.decode(token_ids)
    return tokenized, decoded


def main():
    language = "en"
    whitespace = True
    model_prefix = f"sentencepiece_{language}_suffix_{whitespace}"
    train_sentencepiece(model_prefix=model_prefix, whitespace=whitespace)

    test_text = "He attended Fairfield Seminary and graduated from Union College with a degree in civil engineering in 1846, where he was a member of the Kappa Alpha Society and was elected to Phi Beta Kappa.  After college he taught school while studying law and attained admission to the bar in 1848.  He practiced in Fonda and Broadalbin, and relocated to Johnstown in 1862."
    encoded, decoded = test_tokenizer(model_prefix=model_prefix, test_text=test_text)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}<")
    tokenizer_to_huggingface(model_prefix=model_prefix)
    encoded, decoded = test_tokenizer_hf(model_prefix=model_prefix, test_text=test_text)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}<")


if __name__ == "__main__":
    main()
