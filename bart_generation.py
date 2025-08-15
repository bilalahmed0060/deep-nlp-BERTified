import argparse
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from torch.nn import functional as F
from torch.utils.data import Dataset  # add this alongside DataLoader, TensorDataset


TQDM_DISABLE = False

# ==== RL helpers (self-critical training) ====

def compute_penalized_bleu(inputs, refs, hyps):
    """
    inputs, refs, hyps: lists of strings (same length).
    Returns a single float = penalized BLEU used as reward.
    """
    bleu = BLEU(effective_order=True)  # Use same BLEU flavor as RL
    bleu_ref = bleu.corpus_score(refs, [hyps]).score
    bleu_in = 100 - bleu.corpus_score(inputs, [hyps]).score
    return (bleu_ref * bleu_in) / 52.0


@torch.no_grad()
def greedy_decode(model, tokenizer, input_ids, attention_mask, max_len=50, device="cpu"):
    was_training = model.training
    model.eval()
    out = model.generate(input_ids=input_ids.to(device),
                         attention_mask=attention_mask.to(device),
                         max_length=max_len,
                         num_beams=1, do_sample=False, early_stopping=False,
                         no_repeat_ngram_size=2, pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=tokenizer.eos_token_id)
    if was_training: model.train()
    return out


@torch.no_grad()

def sample_decode(model, tokenizer, input_ids, attention_mask, max_len=50, device="cpu"):
    was_training = model.training
    model.eval()
    out = model.generate(input_ids=input_ids.to(device),
                         attention_mask=attention_mask.to(device),
                         max_length=max_len, min_length=8,
                         do_sample=True, top_p=0.9, top_k=50, temperature=1.5,
                         early_stopping=False, repetition_penalty=1.1,
                         num_return_sequences=1,
                         pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=tokenizer.eos_token_id)
    if was_training: model.train()
    return out



def sequence_logprob(model, input_ids, attention_mask, generated_ids, tokenizer, device="cpu"):
    """
    Compute log p(y|x) for a generated sequence y using teacher-forced next-token logprobs.
    generated_ids: [B, T]
    """
    was_training = model.training
    model.train()  # we need grads for policy gradient
    
    decoder_input_ids = generated_ids[:, :-1].to(device)
    labels = generated_ids[:, 1:].clone().to(device)

    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
    )
    # [B, T-1, V]
    log_probs = torch.log_softmax(outputs.logits, dim=-1)
    # gather token logprobs for the target sequence
    token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    pad_id = tokenizer.pad_token_id
    mask = (labels != pad_id).float()
    seq_logprob = (token_lp * mask).sum(dim=-1)  # [B]
    
    # Restore original training state
    if not was_training:
        model.eval()
    
    return seq_logprob


class ParaphraseRLDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        sep = tokenizer.sep_token if tokenizer.sep_token is not None else '</s>'

        # raw inputs for penalty term
        self.inputs_raw = self.df["sentence1"].tolist()

        # augmented inputs for encoding (same as supervised)
        self.inputs_text = [
            f"{row['sentence1']}{sep}{row.get('sentence1_segment_location','')}{sep}{row.get('paraphrase_type_ids','')}"
            for _, row in self.df.iterrows()
        ]
        self.refs_text = self.df['sentence2'].tolist() if 'sentence2' in self.df.columns else [""] * len(self.df)

        enc = tokenizer(
            self.inputs_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        self.input_ids = enc.input_ids
        self.attention_mask = enc.attention_mask

    def __len__(self):
        # DataLoader's sampler calls this
        return int(self.df.shape[0])

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.inputs_text[idx],  # not used for reward; only for clarity
            self.refs_text[idx],
            self.inputs_raw[idx],   # raw sentence1 for the penalty term
        )


_bleu = BLEU(effective_order=True) 

def penalized_bleu_per_sample(inputs_raw, refs, hyps):
    scores = []
    for s_in, s_ref, s_h in zip(inputs_raw, refs, hyps):
        ref_bleu = _bleu.sentence_score(s_ref, [s_h]).score
        in_bleu  = _bleu.sentence_score(s_in,  [s_h]).score
        scores.append(ref_bleu * (100 - in_bleu) / 52.0)
    return scores


def rl_transform_data(dataset_df, tokenizer, max_length=256, batch_size=8, shuffle=True):
    ds = ParaphraseRLDataset(dataset_df, tokenizer, max_length=max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def rl_finetune_epoch(model, rl_loader, tokenizer, device, lr=2e-6, max_len=50):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    bar = tqdm(rl_loader, disable=TQDM_DISABLE)
    for batch in bar:
        input_ids, attention_mask, inputs_text, refs_text, inputs_raw = batch
        if isinstance(refs_text, tuple): refs_text = list(refs_text)
        if isinstance(inputs_raw, tuple): inputs_raw = list(inputs_raw)

        with torch.no_grad():
            greedy_ids = greedy_decode(model, tokenizer, input_ids, attention_mask, max_len, device)
            sample_ids = sample_decode(model, tokenizer, input_ids, attention_mask, max_len, device)

            greedy_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in greedy_ids]
            sample_text = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in sample_ids]

            r_sample_list = penalized_bleu_per_sample(inputs_raw, refs_text, sample_text)
            r_greedy_list = penalized_bleu_per_sample(inputs_raw, refs_text, greedy_text)

        r_sample = torch.tensor(r_sample_list, dtype=torch.float, device=device)
        r_greedy = torch.tensor(r_greedy_list, dtype=torch.float, device=device)
        advantage = (r_sample - r_greedy).detach()
        delta = advantage.std().item()
        
        # --- Policy gradient loss (only if we have variance) ---
        if delta < 1e-6:
            pg_loss = torch.tensor(0.0, device=device)
            advantage_normalized = advantage  # Keep original for display
        else:
            # Don't normalize to zero - use a minimum scale factor
            scale_factor = max(advantage.std().item(), 0.1)  # Minimum scale of 0.1
            advantage_normalized = (advantage - advantage.mean()) / scale_factor
            
            logp = sequence_logprob(
                model,
                input_ids.to(device),
                attention_mask.to(device),
                sample_ids.to(device),
                tokenizer,
                device
            )
            pg_loss = -(advantage_normalized * logp).mean()

        # --- MLE stabilizer on the same batch ---
        target_enc = tokenizer(list(refs_text), truncation=True, padding="max_length",
                               max_length=256, return_tensors="pt")
        labels = target_enc.input_ids.to(device)
        labels[labels == tokenizer.pad_token_id] = -100
        mle_outputs = model(input_ids=input_ids.to(device),
                            attention_mask=attention_mask.to(device),
                            labels=labels)
        mle_loss = mle_outputs.loss

        # --- Combine ---
        lambda_rl = 0.3
        loss = lambda_rl * pg_loss + (1.0 - lambda_rl) * mle_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Format advantages with more precision to avoid -0.000 display
        adv_mean = advantage.mean().item()
        adv_norm_mean = advantage_normalized.mean().item()
        adv_norm_std = advantage_normalized.std().item()  # Show the spread of normalized advantages
        
        bar.set_description(
            f"RL loss {loss.item():.4f} | PG {pg_loss.item():.3f} | MLE {mle_loss.item():.3f} | adv {adv_mean:.6f} | adv_norm_std {adv_norm_std:.6f}"
        )



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
    num_epochs = 12
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
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
    bleu = BLEU(effective_order=True)  # Use same BLEU flavor as RL
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
    # ↓ NEW: turn RL phase on/off and control epochs
    parser.add_argument("--use_rl", action="store_true",
                        help="Enable RL fine-tuning on penalized BLEU after MLE training")
    parser.add_argument("--rl_epochs", type=int, default=1,
                        help="Number of RL fine-tuning epochs")
    args = parser.parse_args()
    return args


def finetune_paraphrase_generation(args):
    
    print(f"[DEBUG] entered finetune_paraphrase_generation | use_gpu={args.use_gpu} use_rl={args.use_rl} rl_epochs={args.rl_epochs}")
    
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
    
    
    # ---- RL fine-tuning (optional; controlled by --use_rl) ----
    if args.use_rl:
        print("Starting RL fine-tuning on penalized BLEU...")
        
        # Ensure model is in training mode and has gradients enabled
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        # Get supervised BLEU score before RL fine-tuning
        print("Evaluating supervised model before RL fine-tuning...")
        try:
            supervised_bleu = evaluate_model(model, dev_data, device, tokenizer)
            print(f"[SUPERVISED] Dev penalized BLEU before RL: {supervised_bleu:.3f}")
        except Exception as e:
            print(f"[SUPERVISED] Eval failed: {e}")
            supervised_bleu = 0.0
        
        best_bleu = supervised_bleu
        best_model_state = model.state_dict()
        print(f"Initial best BLEU: {best_bleu:.3f}")
        
        rl_loader = rl_transform_data(train_data, tokenizer, max_length=256, batch_size=8, shuffle=True)
        for epoch in range(args.rl_epochs):
            print(f"\n=== RL Epoch {epoch + 1}/{args.rl_epochs} ===")
            rl_finetune_epoch(model, rl_loader, tokenizer, device, lr=2e-6, max_len=50)
            
            # Evaluate after each RL epoch
            try:
                current_rl_bleu = evaluate_model(model, dev_data, device, tokenizer)
                print(f"[RL Epoch {epoch + 1}] Dev penalized BLEU: {current_rl_bleu:.3f}")
                
                # Update best model if current is better
                if current_rl_bleu > best_bleu:
                    best_bleu = current_rl_bleu
                    torch.save(model.state_dict(), "best_bart_model.pt")
                    print(f"🎉 New best BLEU: {best_bleu:.3f} (RL Epoch {epoch + 1})")
                    
                else:
                    print(f"Current BLEU: {current_rl_bleu:.3f} (no improvement)")
                    
            except Exception as e:
                print(f"[RL Epoch {epoch + 1}] Eval failed: {e}")
        
        
        # Save the best model (either supervised or RL)
        if best_bleu > supervised_bleu:
            print(f"RL-improved model to (BLEU: {best_bleu:.3f})")

        else: 
            print("RL failed to improve the Bleu score")
            
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    model.load_state_dict(torch.load("best_bart_model.pt"))
    model.to(device)


    # test_ids = test_dataset["id"]
    # test_results = test_model(test_dataloader, test_ids, device, model, tokenizer)
    

    # test_results.to_csv("predictions/bart/etpc-paraphrase-generation-test-output-check.csv", index=False)
    


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)