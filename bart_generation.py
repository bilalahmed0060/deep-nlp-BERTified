import argparse
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration, AdamW

TQDM_DISABLE = False


def transform_data(dataset, max_length=256, shuffle=True, batch_size=8):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase_type_ids.
    Return Data Loader.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    has_sentence2 = 'sentence2' in dataset.columns
    sep = tokenizer.sep_token if tokenizer.sep_token is not None else '</s>'

    input_texts = [
        f"{row['sentence1']}{sep}{row['sentence1_segment_location']}{sep}{row['paraphrase_type_ids']}"
        for _, row in dataset.iterrows()
    ]
    target_texts = [row['sentence2'] if has_sentence2 else "" for _, row in dataset.iterrows()]

    input_encodings = tokenizer(
        input_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    if has_sentence2:
        target_encodings = tokenizer(
            target_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        labels = target_encodings.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
    else:
        labels = torch.zeros_like(input_encodings.input_ids)

    dataset = TensorDataset(
        input_encodings.input_ids,
        input_encodings.attention_mask,
        labels
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )

    return dataloader


def train_model(model, train_data, dev_data, device, tokenizer):
    """
    Train the model. Return and save the model.
    """
    # Training parameters
    learning_rate = 1e-5
    num_epochs = 1
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    best_bleu = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        total_loss = 0
        progress_bar = tqdm(train_data, disable=TQDM_DISABLE)
        
        for batch in progress_bar:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_data)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Evaluate on dev set
        if dev_data is not None:
            bleu_score = evaluate_model(model, dev_data, device, tokenizer)
            print(f"Dev BLEU Score: {bleu_score:.3f}")
            
            # Save best model
            if bleu_score > best_bleu:
                best_bleu = bleu_score
                # Save full model and state dict
                torch.save(model.state_dict(), "best_bart_model.pt")
                print(f"New best model saved with BLEU: {bleu_score:.3f}")
    
    # Load best model
    if best_bleu > 0:
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        model.load_state_dict(torch.load("best_bart_model.pt"))
        model.to(device)

    
    return model


def test_model(test_data, test_ids, device, model, tokenizer):
    """
    Test the model. Generate paraphrases for the given sentences (sentence1) and return the results
    in form of a Pandas dataframe with the columns 'id' and 'Generated_sentence2'.
    The data format in the columns should be the same as in the train dataset.
    Return this dataframe.
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_data, disable=TQDM_DISABLE):
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
                do_sample=False,
                temperature=1.0,
                no_repeat_ngram_size=2
            )
            
            # Decode predictions
            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            
            predictions.extend(pred_text)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'id': test_ids,
        'Generated_sentence2': predictions
    })
    
    return results_df


def evaluate_model(model, test_data, device, tokenizer):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
    test_data is a Pandas Dataframe, the column "sentence1" contains all input sentence and 
    the column "sentence2" contains all target sentences
    """
    model.eval()
    bleu = BLEU()   
    predictions = []

    dataloader = transform_data(test_data, shuffle=False)
    with torch.no_grad():
        for batch in dataloader: 
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )
            
            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            
            predictions.extend(pred_text)

    inputs = test_data["sentence1"].tolist()
    references = test_data["sentence2"].tolist()

    # Calculate BLEU score
    bleu_score_reference = bleu.corpus_score(references, [predictions]).score
    # Penalize BLEU score if its to close to the input
    bleu_score_inputs = 100 - bleu.corpus_score(inputs, [predictions]).score

    print(f"BLEU Score: {bleu_score_reference}", f"Negative BLEU Score with input: {bleu_score_inputs}")
    

    # Penalize BLEU and rescale it to 0-100
    # If you perfectly predict all the targets, you should get an penalized BLEU score of around 52
    penalized_bleu = bleu_score_reference * bleu_score_inputs / 52
    print(f"Penalized BLEU Score: {penalized_bleu}")

    return penalized_bleu


def load_best_model(device):
    """
    Load the best saved model for inference
    """
    try:
        # Try to load the best model
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
        model.load_state_dict(torch.load("best_bart_model.pt"))
        print("Loaded best model successfully")
    except:
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
        print("Loaded original BART model")
    
    model.to(device)
    return model


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
    
    model = load_best_model(device)
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv")
    
    # Split train data 80/20 for train/dev
    train_size = int(0.8 * len(train_dataset))
    train_data = train_dataset[:train_size]
    dev_data = train_dataset[train_size:]

    print(f"Loaded {len(train_dataset)} total samples.")
    print(f"Training samples: {len(train_data)}")
    print(f"Dev samples: {len(dev_data)}")

    train_dataloader = transform_data(train_data)
    dev_dataloader = transform_data(dev_data, shuffle=False)
    test_dataloader = transform_data(test_dataset, shuffle=False)

    model = train_model(model, train_dataloader, dev_data, device, tokenizer)

    print("Training finished.")


    test_ids = test_dataset["id"]
    test_results = test_model(test_dataloader, test_ids, device, model, tokenizer)
    

    test_results.to_csv("predictions/bart/etpc-paraphrase-generation-test-output-check.csv", index=False)
    


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)