import argparse
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from optimizer import AdamW


TQDM_DISABLE = False


def transform_data(dataset, max_length=256, shuffle=True, batch_size=8):
    """
    Tokenize and encode the dataset for BART.
    For train data: sentence1 + [SEP] + segment_location
    For test data: sentence1 only (no segment_location column)
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)
    input_ids_list = []
    attention_masks = []
    labels = []

    has_labels = "sentence2" in dataset.columns  # Test data does not have labels
    has_segment_location = "sentence1_segment_location" in dataset.columns

    for _, row in dataset.iterrows():
        # Prepare input text
        if has_segment_location:
            input_text = (
                str(row["sentence1"])
                + tokenizer.sep_token
                + str(row["sentence1_segment_location"])
            )
        else:
            input_text = str(row["sentence1"])

        # Tokenize input
        enc_input = tokenizer(
            input_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids_list.append(enc_input["input_ids"].squeeze(0))
        attention_masks.append(enc_input["attention_mask"].squeeze(0))

        # Tokenize target sentence if available
        if has_labels:
            enc_label = tokenizer(
                row["sentence2"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels.append(enc_label["input_ids"].squeeze(0))

    input_ids = torch.stack(input_ids_list)
    attention_masks = torch.stack(attention_masks)

    if has_labels:
        labels = torch.stack(labels)
        tensor_dataset = TensorDataset(input_ids, attention_masks, labels)
    else:
        tensor_dataset = TensorDataset(input_ids, attention_masks)

    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_data, device, epochs=5, learning_rate=1e-5):
    """
    Fine-tune BART on the training dataset.
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_data, desc=f"Training Epoch {epoch+1}", disable=TQDM_DISABLE):
            input_ids, attention_mask, labels = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device)
            )

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"bart_generation_epoch_{epoch+1}.pt")

    print("Training complete.")
    return model


def evaluate_model(model, dev_data, device, tokenizer):
    """
    Evaluate the model's BLEU score on the dev set.
    """
    model.eval()
    bleu = BLEU()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(dev_data, desc="Evaluating BLEU", disable=TQDM_DISABLE):
            input_ids, attention_mask, labels = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )

            decoded_preds = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            decoded_refs = tokenizer.batch_decode(
                labels,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            predictions.extend(decoded_preds)
            references.extend(decoded_refs)

    # Compute BLEU score
    bleu_score = bleu.corpus_score(predictions, [references]).score
    print(f"\n✅ Baseline BLEU Score on dev set: {bleu_score:.2f}")
    return bleu_score


def test_model(test_data, test_ids, device, model, tokenizer):
    """
    Generate paraphrases for the test set and return as a dataframe.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Generating paraphrases", disable=TQDM_DISABLE):
            input_ids, attention_mask = (
                batch[0].to(device),
                batch[1].to(device),
            )

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )

            decoded_preds = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            predictions.extend(decoded_preds)

    # Build dataframe for submission
    df = pd.DataFrame({
        "id": test_ids,
        "Generated_sentence2": predictions
    })
    return df


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    return args


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    dataset = pd.read_csv("data/etpc-paraphrase-train.csv")

    # Split into train/dev
    train_split = dataset.sample(frac=0.8, random_state=42)
    dev_split = dataset.drop(train_split.index)

    # Create DataLoaders
    train_data = transform_data(train_split, batch_size=8)
    dev_data = transform_data(dev_split, shuffle=False, batch_size=8)

    print(f"Training samples: {len(train_split)}, Dev samples: {len(dev_split)}")

    model = train_model(model, train_data, device)

    # Evaluate BLEU on dev set
    bleu_score = evaluate_model(model, dev_data, device, tokenizer)
    print(f"✅ Baseline BLEU on dev set: {bleu_score:.2f}")

    # Generate predictions for test set
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv")
    test_data = transform_data(test_dataset, shuffle=False, batch_size=8)

    test_ids = test_dataset["id"]
    test_results = test_model(test_data, test_ids, device, model, tokenizer)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False
    )
    print("📦 Test predictions saved to predictions/bart/etpc-paraphrase-generation-test-output.csv")


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
