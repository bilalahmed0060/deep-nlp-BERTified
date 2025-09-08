"""
End-to-end script for training and fine-tuning a BART paraphrase generation model.

This script supports two training modes:
- Supervised MLE training (optional)
- Reinforcement Learning (RL) fine-tuning using self-critical sequence training (SCST),
  with an optional quality-guided reward that blends semantic, syntactic, and lexical signals.

It also includes utilities for evaluation, hyperparameter optimization (via Optuna), and
reloading the best checkpoints for downstream inference.

Usage (examples):
- Supervised only:
    python test_bart.py --do_supervised --supervised_epochs 5
- RL (SCST) fine-tuning only:
    python test_bart.py --use_rl --rl_epochs 2
- RL with quality-guided reward weights (semantic, syntactic, lexical):
    python test_bart.py --use_rl --quality_weights "0.7 0.2 0.1"
"""

import argparse
import json
import os
import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import Optimizer
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from torch.utils.data import Dataset  # add this alongside DataLoader, TensorDataset

TQDM_DISABLE = False

# ==== RL helpers (decoding, logprob, dataset) =================================

@torch.no_grad()
def greedy_decode(model, tokenizer, input_ids, attention_mask, max_len=50, device="cpu"):
    """Deterministic decoding: generates one best sequence per input using greedy search.

    - Disables dropout for generation and restores training mode afterward.
    - Uses typical generation safeguards like `no_repeat_ngram_size` to reduce loops.
    """
    was_training = model.training
    model.eval()
    out = model.generate(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        max_length=max_len,
        num_beams=1, do_sample=False, early_stopping=False,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    if was_training:
        model.train()
    return out


@torch.no_grad()
def sample_decode(model, tokenizer, input_ids, attention_mask, max_len=50, device="cpu"):
    """Stochastic decoding: samples a single sequence per input.

    - Uses top-p (nucleus) and top-k sampling with temperature for diversity.
    - Helpful for RL where exploration is needed.
    """
    was_training = model.training
    model.eval()
    out = model.generate(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        max_length=max_len, min_length=8,
        do_sample=True, top_p=0.9, top_k=50, temperature=1.5,
        early_stopping=False, repetition_penalty=1.1,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    if was_training:
        model.train()
    return out


def sequence_logprob(model, input_ids, attention_mask, generated_ids, tokenizer, device="cpu"):
    """Compute log p(y | x) for generated sequences using teacher forcing.

    - Shifts generated ids to build decoder inputs and labels
    - Masks PAD tokens in the label side
    - Returns per-sample sequence log-probabilities (shape: [batch])
    """
    was_training = model.training
    model.train()  # need grads for policy gradient

    decoder_input_ids = generated_ids[:, :-1].to(device)
    labels = generated_ids[:, 1:].clone().to(device)

    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
    )
    log_probs = torch.log_softmax(outputs.logits, dim=-1)  # [B, T-1, V]
    token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    pad_id = tokenizer.pad_token_id
    mask = (labels != pad_id).float()
    seq_logprob = (token_lp * mask).sum(dim=-1)  # [B]

    if not was_training:
        model.eval()
    return seq_logprob


class ParaphraseRLDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        """Dataset for RL fine-tuning that stores raw inputs for reward computation.

        The model receives the same augmented inputs as in supervised training, while
        reward functions can access the unmodified input sentence as well as the
        reference paraphrase text.
        """
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
        return int(self.df.shape[0])

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.inputs_text[idx],
            self.refs_text[idx],
            self.inputs_raw[idx],
        )


# ==== Reward functions ========================================================

_bleu = BLEU(effective_order=True)

def penalized_bleu_per_sample(inputs_raw: List[str], refs: List[str], hyps: List[str]) -> List[float]:
    """Self-critical baseline reward: BLEU(ref, hyp) lightly penalized by BLEU(input, hyp).

    Encourages semantic similarity to the reference while discouraging copying the input.
    Returns a list of floats in a convenient ~0..100 range.
    """
    scores = []
    for s_in, s_ref, s_h in zip(inputs_raw, refs, hyps):
        ref_bleu = _bleu.sentence_score(s_ref, [s_h]).score
        in_bleu  = _bleu.sentence_score(s_in,  [s_h]).score
        scores.append(ref_bleu * (100 - in_bleu) / 52.0)
    return scores


def quality_guided_reward(inputs_raw: List[str], refs: List[str], hyps: List[str],
                          quality_weights: Tuple[float, float, float]) -> List[float]:
    """Quality-guided reward blended from semantic, syntactic, and lexical signals.

    - Semantic: BLEU(ref, hyp) proxy
    - Syntactic: relative length difference proxy
    - Lexical: overlap-based variation score
    A gentle penalty for input similarity is applied at the end.
    Returns scores in approximately 0..100.
    """
    w_sem, w_syn, w_lex = quality_weights
    scores = []
    for s_in, s_ref, s_h in zip(inputs_raw, refs, hyps):
        # 1) Semantic similarity (proxy)
        semantic_sim = _bleu.sentence_score(s_ref, [s_h]).score / 100.0

        # 2) Syntactic variation (length diff proxy)
        input_words = max(1, len(s_in.split()))
        hyp_words   = len(s_h.split())
        length_diff = abs(input_words - hyp_words)
        length_var = 0.7 if length_diff == 0 else min(1.0, 0.7 + (length_diff / input_words) * 0.3)

        # 3) Lexical variation (overlap penalty)
        input_set = set(s_in.lower().split())
        hyp_set = set(s_h.lower().split())
        common_words = input_set.intersection(hyp_set)
        lexical_var = 1.0 if len(common_words) == 0 else max(0.6, 1.0 - (len(common_words) * 0.05))

        quality_score = w_sem * semantic_sim + w_syn * length_var + w_lex * lexical_var

        # Gentle penalty for input similarity
        input_sim = _bleu.sentence_score(s_in, [s_h]).score / 100.0
        penalty = max(0.7, 1.0 - (input_sim * 0.2))

        final_reward = quality_score * penalty
        scores.append(final_reward * 100.0)
    return scores


# ==== Dataloaders =============================================================

def rl_transform_data(dataset_df, tokenizer, max_length=256, batch_size=8, shuffle=True):
    """Build a dataloader for RL training with access to raw inputs and refs."""
    ds = ParaphraseRLDataset(dataset_df, tokenizer, max_length=max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def transform_data(dataset, max_length=256, shuffle=True, batch_size=8):
    """Tokenize and format data for supervised training.

    Input format:
      sentence1 + SEP + sentence1_segment_location + SEP + paraphrase_type_ids
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    has_sentence2 = 'sentence2' in dataset.columns
    sep = tokenizer.sep_token if tokenizer.sep_token is not None else '</s>'

    input_texts = [
        f"{row['sentence1']}{sep}{row['sentence1_segment_location']}{sep}{row['paraphrase_type_ids']}"
        for _, row in dataset.iterrows()
    ]
    target_texts = [row['sentence2'] if has_sentence2 else "" for _, row in dataset.iterrows()]

    input_encodings = tokenizer(
        input_texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    if has_sentence2:
        target_encodings = tokenizer(
            target_texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        labels = target_encodings.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
    else:
        labels = torch.zeros_like(input_encodings.input_ids)

    dataset = TensorDataset(input_encodings.input_ids, input_encodings.attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return dataloader


# ==== Training / Evaluation ===================================================

def rl_finetune_epoch(
    model,
    rl_loader,
    tokenizer,
    device,
    lr=2e-6,
    max_len=50,
    lambda_rl: float = 0.3,
    rewarder: Optional[Callable[[List[str], List[str], List[str]], List[float]]] = None,
    optimizer: Optional[Optimizer] = None
):
    """Run a single epoch of RL fine-tuning using self-critical policy gradient.

    The rewarder is any callable of the form (inputs_raw, refs, hyps) -> List[float].
    """
    assert rewarder is not None, "A rewarder function must be provided."
    model.train()
    if optimizer is None:
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

            r_sample_list = rewarder(inputs_raw, refs_text, sample_text)
            r_greedy_list = rewarder(inputs_raw, refs_text, greedy_text)

        r_sample = torch.tensor(r_sample_list, dtype=torch.float, device=device)
        r_greedy = torch.tensor(r_greedy_list, dtype=torch.float, device=device)
        advantage = (r_sample - r_greedy).detach()
        delta = advantage.std().item()

        # Policy gradient loss: encourage sampled sequences that beat greedy baseline
        if delta < 1e-6:
            pg_loss = torch.tensor(0.0, device=device)
            advantage_normalized = advantage
        else:
            scale_factor = max(advantage.std().item(), 0.1)
            advantage_normalized = (advantage - advantage.mean()) / scale_factor

            logp = sequence_logprob(
                model, input_ids.to(device), attention_mask.to(device), sample_ids.to(device), tokenizer, device
            )
            pg_loss = -(advantage_normalized * logp).mean()

        # MLE stabilizer: keep the model grounded to references during RL
        target_enc = tokenizer(list(refs_text), truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        labels = target_enc.input_ids.to(device)
        labels[labels == tokenizer.pad_token_id] = -100
        mle_outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels)
        mle_loss = mle_outputs.loss

        # Combine losses using lambda_rl
        loss = lambda_rl * pg_loss + (1.0 - lambda_rl) * mle_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        adv_mean = advantage.mean().item()
        adv_norm_std = advantage_normalized.std().item()
        bar.set_description(
            f"RL loss {loss.item():.4f} | PG {pg_loss.item():.3f} | MLE {mle_loss.item():.3f} | adv {adv_mean:.6f} | adv_norm_std {adv_norm_std:.6f}"
        )


def train_model(model, train_data, dev_data, device, tokenizer, num_epochs=10, lr=1e-5):
    """Run supervised MLE training for a configurable number of epochs.

    Saves the best checkpoint by dev BLEU and returns the (re)loaded model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_data)
        print(f"Average loss: {avg_loss:.4f}")

        if dev_data is not None:
            bleu_score = evaluate_model(model, dev_data, device, tokenizer)
            print(f"Dev BLEU Score: {bleu_score:.2f}")
            if bleu_score > best_bleu:
                best_bleu = bleu_score
                torch.save(model.state_dict(), "best_supervised_bart_model.pt")
                print(f"New best model saved with BLEU: {bleu_score:.2f}")

    if best_bleu > 0:
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        model.load_state_dict(torch.load("best_supervised_bart_model.pt"))
        model.to(device)

    return model


def test_model(test_data, test_ids, device, model, tokenizer):
    """Generate paraphrases for a test dataloader and return a results DataFrame."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_data, disable=TQDM_DISABLE):
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
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
            pred_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in outputs]
            predictions.extend(pred_text)

    results_df = pd.DataFrame({'id': test_ids, 'Generated_sentence2': predictions})
    return results_df


def evaluate_model(model, test_data, device, tokenizer):
    """Evaluate on a DataFrame split and return a penalized BLEU summary score.

    Penalized BLEU combines BLEU(reference, prediction) with a negative BLEU(input, prediction)
    factor to discourage copying the source.
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
            outputs = model.generate(
                input_ids, attention_mask=attention_mask, max_length=50, num_beams=5, early_stopping=True
            )
            pred_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in outputs]
            predictions.extend(pred_text)

    inputs = test_data["sentence1"].tolist()
    references = test_data["sentence2"].tolist()
    bleu_score_reference = bleu.corpus_score(references, [predictions]).score
    bleu_score_inputs = 100 - bleu.corpus_score(inputs, [predictions]).score
    print(f"BLEU Score: {bleu_score_reference:.2f}", f"Negative BLEU Score with input: {bleu_score_inputs:.2f}")

    penalized_bleu = bleu_score_reference * bleu_score_inputs / 52
    print(f"Penalized BLEU Score: {penalized_bleu:.2f}")
    return penalized_bleu


def load_best_model(device):
    """Load the best saved model if available; otherwise load base BART weights.

    Tries local caches first to avoid unnecessary network calls.
    """
    try:
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
        model.load_state_dict(torch.load("best_supervised_bart_model.pt", map_location=device))
        print("Loaded best supervised model successfully")
    except Exception:
        try:
            model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
            print("Loaded original BART model (local cache)")
        except Exception:
            model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
            print("Loaded original BART model (from hub)")
    model.to(device)
    return model


# ==== Utilities / CLI =========================================================

def seed_everything(seed=11711):
    """Make results reproducible across Python, NumPy, and PyTorch (CPU/GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_quality_weights(arg: Optional[str]) -> Optional[Tuple[float, float, float]]:
    """Parse user-provided quality weights for the reward function.

    Accepted formats:
    - None  -> returns None (use SCST reward)
    - "0.7 0.2 0.1" or "0.7,0.2,0.1"
    - Path to a JSON file with keys: semantic, syntactic, lexical
    """
    if arg is None:
        return None
    txt = arg.strip()
    if os.path.isfile(txt):
        with open(txt) as f:
            cfg = json.load(f)
        return (float(cfg["semantic"]), float(cfg["syntactic"]), float(cfg["lexical"]))
    txt = txt.replace(",", " ")
    parts = [p for p in txt.split() if p]
    if len(parts) != 3:
        raise ValueError("--quality_weights must be 3 floats or a JSON file path with semantic/syntactic/lexical.")
    w_sem, w_syn, w_lex = map(float, parts)
    return (w_sem, w_syn, w_lex)


def get_args():
    """Define and parse all CLI arguments for training and evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")

    # RL control
    parser.add_argument("--use_rl", action="store_true",
                        help="Enable RL fine-tuning after (optional) MLE training")
    parser.add_argument("--rl_epochs", type=int, default=2,
                        help="Number of RL fine-tuning epochs (default 2 to match your SCST call)")

    # Optional supervised phase (off by default to match your SCST script)
    parser.add_argument("--do_supervised", action="store_true",
                        help="Run MLE supervised training before RL (default: off)")

    # Switch key: presence triggers quality-guided mode
    parser.add_argument("--quality_weights", type=str, default=None,
                        help="Either 'sem syn lex' floats, 'sem,syn,lex', or a JSON file path.")
    
    parser.add_argument("--supervised_epochs", type=int, default=10,
                        help="Number of supervised (MLE) epochs when --do_supervised is set.")

    # RL knobs (also used by HPO defaults)
    parser.add_argument("--rl_lr", type=float, default=2e-6,
                        help="Learning rate for RL fine-tuning (default: 2e-6)")
    parser.add_argument("--rl_lambda", type=float, default=0.3,
                        help="Weight for RL vs MLE loss blend (default: 0.3)")
    parser.add_argument("--rl_max_len", type=int, default=50,
                        help="Max generation length during RL (default: 50)")
    parser.add_argument("--rl_batch_size", type=int, default=8,
                        help="Batch size for RL fine-tuning (default: 8)")

    # Hyperparameter optimization
    parser.add_argument("--hpo", action="store_true", help="Run hyperparameter optimization before training")
    parser.add_argument("--hpo_trials", type=int, default=20, help="Number of HPO trials (default: 20)")
    parser.add_argument("--hpo_timeout", type=int, default=None,
                        help="Optional timeout for HPO in seconds (default: None)")
    parser.add_argument("--hpo_rl_epochs", type=int, default=1,
                        help="RL epochs per HPO trial (keep small for speed; default: 1)")

    return parser.parse_args()


# ==== Main orchestration ======================================================

def finetune_paraphrase_generation(args):
    """Main orchestration: optionally run supervised training, optional HPO, and RL fine-tuning.

    At the end, reload the best available checkpoint and (optionally) run test-time generation.
    """
    print(f"[DEBUG] entered finetune_paraphrase_generation | use_gpu={args.use_gpu} use_rl={args.use_rl} rl_epochs={args.rl_epochs} quality_weights={args.quality_weights}")

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = load_best_model(device)

    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset  = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv")

    # Split 80/20
    train_size = int(0.8 * len(train_dataset))
    train_data = train_dataset[:train_size]
    dev_data   = train_dataset[train_size:]

    print(f"Loaded {len(train_dataset)} total samples.")
    print(f"Training samples: {len(train_data)}")
    print(f"Dev samples: {len(dev_data)}")

    # Supervised (optional; default off)
    if args.do_supervised:
        train_dataloader = transform_data(train_data)
        model = train_model(model, train_dataloader, dev_data, device, tokenizer,
                        num_epochs=args.supervised_epochs)
        
        print("Supervised training finished.")

    # ---- Optional Hyperparameter Optimization (RL hyperparameters) ----
    if args.hpo and args.use_rl:
        print("\n[HPO] Starting hyperparameter optimization for RL fine-tuning...")
        try:
            import optuna
        except Exception as e:
            raise RuntimeError("Optuna is required for --hpo. Please install it (e.g., pip install optuna)") from e

        # Cache shared objects
        base_state_dict = model.state_dict()  # snapshot to reset per trial

        def objective(trial):
            # Suggest hyperparameters
            rl_lr = trial.suggest_float("rl_lr", 1e-7, 5e-5, log=True)
            lambda_rl = trial.suggest_float("lambda_rl", 0.1, 0.7)
            rl_max_len = trial.suggest_int("rl_max_len", 40, 80, step=10)
            rl_batch_size = trial.suggest_categorical("rl_batch_size", [4, 8, 16])

            # Reset model to base weights for each trial
            model.load_state_dict(base_state_dict)
            model.to(device)
            model.train()

            # Rewarder selection
            qw = parse_quality_weights(args.quality_weights)
            if qw is not None:
                rewarder = lambda inputs_raw, refs_text, hyps: quality_guided_reward(inputs_raw, refs_text, hyps, qw)
            else:
                rewarder = penalized_bleu_per_sample

            # Build loader for this trial (can subsample to speed up)
            # Use a deterministic subset for fairness across trials
            subset_frac = 0.5 if len(train_data) > 1000 else 1.0
            if subset_frac < 1.0:
                subset_size = int(len(train_data) * subset_frac)
                trial_train = train_data.sample(n=subset_size, random_state=args.seed)
            else:
                trial_train = train_data

            rl_loader = rl_transform_data(trial_train, tokenizer, max_length=256, batch_size=rl_batch_size, shuffle=True)

            optimizer = torch.optim.AdamW(model.parameters(), lr=rl_lr)

            # Short RL fine-tuning for this trial
            for _ in range(max(1, args.hpo_rl_epochs)):
                rl_finetune_epoch(
                    model,
                    rl_loader,
                    tokenizer,
                    device,
                    lr=rl_lr,
                    max_len=rl_max_len,
                    lambda_rl=lambda_rl,
                    rewarder=rewarder,
                    optimizer=optimizer,
                )

            # Evaluate on the same dev split
            try:
                score = evaluate_model(model, dev_data, device, tokenizer)
            except Exception as eval_err:
                # Penalize failed trials
                print(f"[HPO] Eval failed in trial: {eval_err}")
                return -1e9

            return float(score)

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=args.hpo_trials, timeout=args.hpo_timeout)
        print(f"[HPO] Best value: {study.best_value:.4f}")
        print(f"[HPO] Best params: {study.best_trial.params}")

        # Reset model to the base supervised weights before final RL run
        model.load_state_dict(base_state_dict)

        # Apply best params to args for final training
        best = study.best_trial.params
        args.rl_lr = float(best.get("rl_lr", args.rl_lr))
        args.rl_lambda = float(best.get("lambda_rl", args.rl_lambda))
        args.rl_max_len = int(best.get("rl_max_len", args.rl_max_len))
        args.rl_batch_size = int(best.get("rl_batch_size", args.rl_batch_size))

        # Ensure RL begins from the best supervised checkpoint on disk, if available
        try:
            if os.path.exists("best_supervised_bart_model.pt"):
                model.load_state_dict(torch.load("best_supervised_bart_model.pt", map_location=device))
                print("[HPO] Reloaded best supervised checkpoint before final RL run.")
            else:
                print("[HPO] No best supervised checkpoint found; proceeding with base weights.")
        except Exception as e:
            print(f"[HPO] Failed to reload best supervised checkpoint: {e}. Proceeding with base weights.")

    # ---- RL fine-tuning (optional) ----
    if args.use_rl:
        # Decide rewarder
        qw = parse_quality_weights(args.quality_weights)
        if qw is not None:
            print(f"[Mode] Quality-Guided RL with weights (sem, syn, lex) = {qw}")
            rewarder = lambda inputs_raw, refs_text, hyps: quality_guided_reward(inputs_raw, refs_text, hyps, qw)
        else:
            print("[Mode] SCST RL (penalized BLEU)")
            rewarder = penalized_bleu_per_sample

        # Ensure trainable
        model.train()
        for p in model.parameters():
            p.requires_grad = True
            
        rl_lr = args.rl_lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=rl_lr)
        
        
        # Pre-RL evaluation
        print("Evaluating model before RL fine-tuning...")
        try:
            pre_bleu = evaluate_model(model, dev_data, device, tokenizer)
            print(f"[PRE] Dev penalized BLEU before RL: {pre_bleu:.2f}")
        except Exception as e:
            print(f"[PRE] Eval failed: {e}")
            pre_bleu = 0.0

        best_bleu = pre_bleu
        rl_loader = rl_transform_data(train_data, tokenizer, max_length=256, batch_size=args.rl_batch_size, shuffle=True)

        for epoch in range(args.rl_epochs):
            print(f"\n=== RL Epoch {epoch + 1}/{args.rl_epochs} ===")
            rl_finetune_epoch(
                model,
                rl_loader,
                tokenizer,
                device,
                lr=rl_lr,
                max_len=args.rl_max_len,
                lambda_rl=args.rl_lambda,
                rewarder=rewarder,
                optimizer=optimizer,
            )

            # Evaluate
            try:
                current_bleu = evaluate_model(model, dev_data, device, tokenizer)
                print(f"[RL Epoch {epoch + 1}] Dev penalized BLEU: {current_bleu:.2f}")
                if current_bleu > best_bleu:
                    best_bleu = current_bleu
                    torch.save(model.state_dict(), "best_bart_rl_model.pt")
                    print(f"New best BLEU: {best_bleu:.2f} (saved)")
                else:
                    print("No improvement this epoch.")
            except Exception as e:
                print(f"[RL Epoch {epoch + 1}] Eval failed: {e}")

        if best_bleu > pre_bleu:
            print(f"RL improved BLEU: {pre_bleu:.2f} → {best_bleu:.2f}")
        else:
            print("RL did not improve BLEU.")

    #Reload best (if any)
    
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    try:
        if os.path.exists("best_bart_rl_model.pt"):
            model.load_state_dict(torch.load("best_bart_rl_model.pt", map_location=device))
            print("Reloaded best RL model for downstream inference.")
        elif os.path.exists("best_supervised_bart_model.pt"):
            model.load_state_dict(torch.load("best_supervised_bart_model.pt", map_location=device))
            print("Reloaded best supervised model for downstream inference.")
        else:
            print("No best model on disk; using base pretrained weights.")
    except (RuntimeError, FileNotFoundError) as e:
        print(f"Could not load saved model weights: {e}")
        print("Using base pretrained weights instead.")

    model.to(device)
    
    print("\nGenerating predictions on test set...")

    test_ids = test_dataset["id"]
    test_dataloader = transform_data(test_dataset, shuffle=False)
    test_results = test_model(test_dataloader, test_ids, device, model, tokenizer)
    test_results.to_csv("predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
