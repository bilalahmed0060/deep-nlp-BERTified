import argparse
import ast
import random
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
from sklearn.metrics import matthews_corrcoef

# --- Optimizer import (keeps your local optimizer.py, falls back otherwise) ---
try:
    from optimizer import AdamW  # your local optimizer module
except Exception:
    try:
        from torch.optim import AdamW  # fallback
    except Exception:
        from transformers.optimization import AdamW  # last resort

TQDM_DISABLE = False


# =========================
# Model: BART + masked-mean pooling (returns LOGITS)
# =========================
class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=26, dropout=0.1):
        super().__init__()
        self.bart = BartModel.from_pretrained("facebook/bart-large")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, H]

        if attention_mask is None:
            pooled = last_hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()      # [B, T, 1]
            summed = (last_hidden * mask).sum(dim=1)         # [B, H]
            lengths = torch.clamp(mask.sum(dim=1), min=1.0)  # [B, 1]
            pooled = summed / lengths

        logits = self.classifier(self.dropout(pooled))       # [B, C]
        return logits


# =========================
# Asymmetric Loss (ASL)
# =========================
class AsymmetricLossMultiLabel(nn.Module):
    """
    Ridnik et al., 2021 — decoupled focusing for pos/neg + easy-negative clipping.
    """
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
        disable_torch_grad_focal: bool = True,
    ):
        super().__init__()
        assert reduction in {"mean", "sum", "none"}
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self.disable_torch_grad_focal = disable_torch_grad_focal

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)
        xs_neg = torch.sigmoid(-logits)  # 1 - sigmoid(logits)

        # clip easy negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = torch.clamp(xs_neg + self.clip, max=1.0)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg  # log-likelihood

        pt = targets * xs_pos + (1.0 - targets) * xs_neg
        one_minus_pt = 1.0 - pt
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)

        if self.disable_torch_grad_focal:
            focal_weight = (one_minus_pt.clamp(min=self.eps)) ** gamma.detach()
        else:
            focal_weight = (one_minus_pt.clamp(min=self.eps)) ** gamma

        loss *= focal_weight
        loss = -loss.mean(dim=1)  # average over C per sample

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # [B]


# =========================
# Data & stats
# =========================
_VALID_TYPES = [i for i in range(1, 32) if i not in {12, 19, 20, 23, 27}]
_TYPE_TO_IDX = {t: i for i, t in enumerate(_VALID_TYPES)}
_NUM_CLASSES = len(_VALID_TYPES)  # 26


def transform_data(dataset: pd.DataFrame, max_length=512, sampler: Optional[WeightedRandomSampler] = None,
                   shuffle: Optional[bool] = None):
    """
    Returns a DataLoader (with labels if present).
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    input_ids_list, attention_masks, labels = [], [], []
    has_labels = "paraphrase_type_ids" in dataset.columns

    for _, row in dataset.iterrows():
        enc = tokenizer(
            row["sentence1"],
            row["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_ids_list.append(enc["input_ids"])
        attention_masks.append(enc["attention_mask"])

        if has_labels:
            types = ast.literal_eval(row["paraphrase_type_ids"])
            label_vec = [0] * _NUM_CLASSES
            for t in types:
                if t in _TYPE_TO_IDX:
                    label_vec[_TYPE_TO_IDX[t]] = 1
            labels.append(label_vec)

    input_ids = torch.tensor(input_ids_list)
    attention_mask = torch.tensor(attention_masks)

    if has_labels:
        labels = torch.tensor(labels, dtype=torch.float)
        tensor_ds = TensorDataset(input_ids, attention_mask, labels)
    else:
        tensor_ds = TensorDataset(input_ids, attention_mask)

    if sampler is not None:
        return DataLoader(tensor_ds, batch_size=32, sampler=sampler)
    return DataLoader(tensor_ds, batch_size=32, shuffle=(has_labels if shuffle is None else shuffle))


def compute_class_priors(train_df: pd.DataFrame) -> np.ndarray:
    counts = np.zeros(_NUM_CLASSES, dtype=np.float64)
    N = len(train_df)
    for _, row in train_df.iterrows():
        types = ast.literal_eval(row["paraphrase_type_ids"])
        for t in types:
            if t in _TYPE_TO_IDX:
                counts[_TYPE_TO_IDX[t]] += 1.0
    p = np.clip(counts / max(N, 1), 1e-6, 1 - 1e-6)  # avoid 0/1
    return p  # [C]


def compute_instance_weights(df: pd.DataFrame) -> np.ndarray:
    """
    per-instance weight = sum of inverse class priors for its labels
    """
    priors = compute_class_priors(df)  # [C]
    inv = 1.0 / priors
    weights = []
    for _, row in df.iterrows():
        types = ast.literal_eval(row["paraphrase_type_ids"])
        w = 0.0
        for t in types:
            if t in _TYPE_TO_IDX:
                w += inv[_TYPE_TO_IDX[t]]
        weights.append(w if w > 0 else inv.mean())
    weights = np.array(weights, dtype=np.float64)
    return weights / weights.mean()


# =========================
# Threshold tuning & eval
# =========================
def tune_thresholds_for_mcc(model: nn.Module, dev_loader: DataLoader, device,
                            num_classes: int = _NUM_CLASSES,
                            grid: Optional[Sequence[float]] = None,
                            prior_logit: Optional[torch.Tensor] = None,
                            tau: float = 0.0):
    """
    Returns thresholds maximizing per-class MCC on dev.
    If prior_logit and tau>0, adjusts logits by -tau*prior_logit before sigmoid.
    """
    if grid is None:
        grid = np.linspace(0.02, 0.98, 49)

    model.eval()
    probs_list, y_list = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device); attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            if prior_logit is not None and tau > 0:
                logits = logits - tau * prior_logit.to(logits.device)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            y_list.append(labels.numpy())

    P = np.vstack(probs_list)            # [N, C]
    Y = np.vstack(y_list).astype(int)    # [N, C]

    thresholds = np.full(num_classes, 0.5, dtype=float)
    for c in range(num_classes):
        best_t, best_mcc = 0.5, -1.0
        y_true = Y[:, c]
        for t in grid:
            pred = (P[:, c] >= t).astype(int)
            mcc = matthews_corrcoef(y_true, pred)
            if np.isnan(mcc):
                mcc = -1.0
            if mcc > best_mcc:
                best_mcc, best_t = mcc, t
        thresholds[c] = best_t
    return thresholds


def evaluate_model(model: nn.Module, dev_loader: DataLoader, device,
                   thresholds: Optional[np.ndarray] = None,
                   prior_logit: Optional[torch.Tensor] = None,
                   tau: float = 0.0):
    """
    Mean per-label accuracy + mean per-label MCC.
    If thresholds given, use them per class. Optionally apply logit adjustment.
    """
    all_pred, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in dev_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device); attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            if prior_logit is not None and tau > 0:
                logits = logits - tau * prior_logit.to(logits.device)
            probs = torch.sigmoid(logits).cpu().numpy()
            if thresholds is None:
                pred = (probs > 0.5).astype(int)
            else:
                pred = (probs >= thresholds).astype(int)
            all_pred.append(pred); all_labels.append(labels.numpy())

    Yhat = np.vstack(all_pred); Y = np.vstack(all_labels)
    accs, mccs = [], []
    for i in range(Y.shape[1]):
        accs.append((Y[:, i] == Yhat[:, i]).mean())
        mccs.append(matthews_corrcoef(Y[:, i], Yhat[:, i]))
    return float(np.mean(accs)), float(np.mean(mccs))


def test_model(model: nn.Module, test_loader: DataLoader, test_ids, device,
               thresholds: Optional[np.ndarray] = None,
               prior_logit: Optional[torch.Tensor] = None,
               tau: float = 0.0):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask = batch[:2]
            input_ids = input_ids.to(device); attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            if prior_logit is not None and tau > 0:
                logits = logits - tau * prior_logit.to(logits.device)
            probs = torch.sigmoid(logits).cpu().numpy()
            if thresholds is None:
                pred = (probs > 0.5).astype(int)
            else:
                pred = (probs >= thresholds).astype(int)
            preds.extend(pred.tolist())
    return pd.DataFrame({"id": test_ids, "Predicted_Paraphrase_Types": preds})


# =========================
# Train loop
# =========================
def train_model(model, train_data, dev_data, device, loss_fn, epochs=5, lr=1e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        train_loss = 0.0
        num_batches = 0
        model.train()
        for batch in tqdm(train_data, disable=TQDM_DISABLE):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        train_loss /= max(1, num_batches)
        acc, matt = evaluate_model(model, dev_data, device)
        print(f"Epoch {epoch+1}: dev accuracy :: {acc:.3f}, dev loss :: {train_loss:.3f}, MCC :: {matt:.3f}")
    return model


# =========================
# Misc
# =========================
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=11711)
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.1)

    # ASL params
    p.add_argument("--gamma_pos", type=float, default=0.0)
    p.add_argument("--gamma_neg", type=float, default=2.0)  # try 2.0 (was 4.0)
    p.add_argument("--clip", type=float, default=0.1)       # try 0.1 (was 0.05)

    # Prior-aware logit adjustment
    p.add_argument("--logit_adjust_tau", type=float, default=0.0, help="0 disables; try 0.25–1.0")

    # Threshold tuning grid
    p.add_argument("--th_grid_start", type=float, default=0.02)
    p.add_argument("--th_grid_end", type=float, default=0.98)
    p.add_argument("--th_grid_steps", type=int, default=49)

    # Weighted sampler
    p.add_argument("--use_weighted_sampler", action="store_true")
    return p.parse_args()


# =========================
# Main
# =========================
def finetune_paraphrase_detection(args):
    seed_everything(args.seed)
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    model = BartWithClassifier(num_labels=_NUM_CLASSES, dropout=args.dropout).to(device)

    train_df = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_df  = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")

    # Split train/dev
    train_split = train_df.sample(frac=0.8, random_state=42).reset_index(drop=True)
    dev_split   = train_df.drop(train_split.index).reset_index(drop=True)

    # Weighted sampler (optional)
    sampler = None
    if args.use_weighted_sampler:
        inst_weights = compute_instance_weights(train_split)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(inst_weights, dtype=torch.double),
            num_samples=len(inst_weights),
            replacement=True,
        )

    # DataLoaders
    train_loader = transform_data(train_split, sampler=sampler, shuffle=(sampler is None))
    dev_loader   = transform_data(dev_split, shuffle=False)
    test_loader  = transform_data(test_df, shuffle=False)

    print(f"Loaded {len(train_df)} training samples.")
    print(f"ASL params -> gamma_pos={args.gamma_pos}, gamma_neg={args.gamma_neg}, clip={args.clip}")
    if args.use_weighted_sampler:
        print("Using WeightedRandomSampler to upsample rare-label instances.")
    if args.logit_adjust_tau > 0:
        print(f"Using logit adjustment with tau={args.logit_adjust_tau}")

    # Loss
    asloss = AsymmetricLossMultiLabel(
        gamma_pos=args.gamma_pos,
        gamma_neg=args.gamma_neg,
        clip=args.clip,
        reduction="mean",
    )

    # Train
    model = train_model(model, train_loader, dev_loader, device, loss_fn=asloss,
                        epochs=args.epochs, lr=args.lr)

    print("Training finished.")

    # --- Prior-aware logit adjustment setup ---
    prior_logit = None
    if args.logit_adjust_tau > 0:
        priors = compute_class_priors(train_split)                 # [C]
        prior_logit_np = np.log(priors / (1.0 - priors))           # log-odds
        prior_logit = torch.from_numpy(prior_logit_np).float()     # torch tensor [C]

    # --- Threshold tuning on dev (after optional logit adjustment) ---
    grid = np.linspace(args.th_grid_start, args.th_grid_end, args.th_grid_steps)
    opt_thresholds = tune_thresholds_for_mcc(
        model, dev_loader, device, num_classes=_NUM_CLASSES,
        grid=grid, prior_logit=prior_logit, tau=args.logit_adjust_tau
    )
    acc_t, mcc_t = evaluate_model(
        model, dev_loader, device, thresholds=opt_thresholds,
        prior_logit=prior_logit, tau=args.logit_adjust_tau
    )
    print(f"Dev (threshold-tuned) accuracy: {acc_t:.3f}  MCC: {mcc_t:.3f}")

    # Final eval (0.5) just for reference
    acc_raw, mcc_raw = evaluate_model(
        model, dev_loader, device, thresholds=None,
        prior_logit=prior_logit, tau=args.logit_adjust_tau
    )
    print(f"Dev (fixed 0.5)   accuracy: {acc_raw:.3f}  MCC: {mcc_raw:.3f}")

    # Test predictions using tuned thresholds (+ optional logit adjustment)
    test_ids = test_df["id"]
    test_results = test_model(
        model, test_loader, test_ids, device,
        thresholds=opt_thresholds, prior_logit=prior_logit, tau=args.logit_adjust_tau
    )
    test_results.to_csv("predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False)
    print("Wrote predictions to predictions/bart/etpc-paraphrase-detection-test-output.csv")


if __name__ == "__main__":
    args = get_args()
    finetune_paraphrase_detection(args)
