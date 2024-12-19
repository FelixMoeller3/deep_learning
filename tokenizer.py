from datasets import load_dataset
import sentencepiece as spm
import pandas as pd
import os
import tempfile


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
        model_prefix=model_prefix,  # Set the model prefix
    )
    # Clean up the temporary file
    os.remove(dataset_file_path)


def test_tokenizer(model_prefix, test_text):
    # Load the trained SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")

    # Encode the test text
    encoded = sp.EncodeAsPieces(test_text)

    # Decode the encoded text
    decoded = sp.DecodePieces(encoded)

    return encoded, decoded


def main():
    language = "en"
    whitespace = True
    model_prefix = f"sentencepiece_{language}_suffix_{whitespace}"
    train_sentencepiece(model_prefix=model_prefix, whitespace=whitespace)

    test_text = (
        "He attended Fairfield Seminary and graduated from Union College with a degree in civil engineering in 1846, where he was a member of the Kappa Alpha Society and was elected to Phi Beta Kappa.  After college he taught school while studying law and attained admission to the bar in 1848.  He practiced in Fonda and Broadalbin, and relocated to Johnstown in 1862."
    )
    encoded, decoded = test_tokenizer(
        model_prefix=model_prefix, test_text=test_text
    )
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")


if __name__ == "__main__":
    main()
