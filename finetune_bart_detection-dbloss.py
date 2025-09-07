import argparse
import ast
import os
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

# Optimizer: prefer local optimizer.AdamW; fallback to torch.optim.AdamW
try:
    from optimizer import AdamW
except Exception:
    from torch.optim import AdamW

# ---- plotting (headless-safe) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TQDM_DISABLE = False

# Label space (drop: 12,19,20,23,27)
_VALID_TYPES = [i for i in range(1, 32) if i not in {12, 19, 20, 23, 27}]
_TYPE_TO_IDX = {t: i for i, t in enumerate(_VALID_TYPES)}
_NUM_CLASSES = len(_VALID_TYPES)

# -------------------------
# Model: BART + masked-mean pooling (returns LOGITS)
# -------------------------
class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=_NUM_CLASSES, dropout=0.1):
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

# -------------------------
# Distribution-Balanced Loss (DBLoss)
# -------------------------
class DBLoss(nn.Module):
    """
    Distribution-Balanced Loss for multi-label:
      - Re-balanced weighting r_hat using inverse class freq per-instance
      - Negative-Tolerant Regularization (NTR): class bias v_i + lambda on negatives
      - Optional focal modulation (gamma); use gamma=0.0 for DB-0FL
    """
    def __init__(
        self,
        class_counts: torch.Tensor,  # [C] positives in TRAIN SPLIT
        alpha: float = 0.1,
        beta: float = 10.0,
        mu: float = 0.9,
        lam: float = 2.0,
        kappa: float = 0.05,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-12,
    ):
        super().__init__()
        assert reduction in {"mean", "sum", "none"}
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.lam = lam
        self.kappa = kappa
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

        counts = class_counts.clone().float().clamp_min(1.0) # avoid zero
        self.register_buffer("class_counts", counts)
        self.register_buffer("inv_counts", 1.0 / counts)

        self.N_instances = None
        self.bias_ready = False
        self.register_buffer("v", torch.zeros_like(counts))

    def set_num_instances(self, N_instances: int):
        self.N_instances = max(int(N_instances), 1)
        p = (self.class_counts / float(self.N_instances)).clamp_(1e-6, 1 - 1e-6)
        b_hat = torch.log(p / (1.0 - p))
        self.v = -self.kappa * b_hat
        self.bias_ready = True

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not self.bias_ready:
            raise RuntimeError("Call DBLoss.set_num_instances(N) before training.")

        B, C = logits.shape
        device = logits.device
        inv_counts = self.inv_counts.to(device)             # [C]
        v = self.v.to(device).view(1, C)                    # [1, C]
        y = targets

        # Re-balanced weighting r_hat (per sample/label)
        pos_mask = (y > 0).float()                          # [B, C]
        denom = torch.matmul(pos_mask, inv_counts)          # [B]
        default_denom = inv_counts.sum()
        denom = torch.where(denom > 0, denom, torch.full_like(denom, default_denom))
        denom = denom.view(B, 1)
        r = inv_counts.view(1, C) / denom                   # [B, C]
        r_hat = self.alpha + torch.sigmoid(self.beta * (r - self.mu))

        # Negative-tolerant regularization (shift + lambda)
        z_shift = logits - v
        q_pos = torch.sigmoid(z_shift)
        q_neg = torch.sigmoid(self.lam * z_shift)

        one_minus_q_pos = 1.0 - q_pos
        one_minus_q_neg = 1.0 - q_neg

        if self.gamma > 0.0:
            # Focalized DB
            pos_term = - (one_minus_q_pos.clamp_min(self.eps) ** self.gamma) * torch.log(q_pos.clamp_min(self.eps))
            neg_term = - (q_neg.clamp_min(self.eps) ** self.gamma) * torch.log(one_minus_q_neg.clamp_min(self.eps))
        else:
            # DB-0FL
            pos_term = - torch.log(q_pos.clamp_min(self.eps))
            neg_term = - torch.log(one_minus_q_neg.clamp_min(self.eps))

        loss_per_elem = r_hat * (y * pos_term + (1.0 - y) * (neg_term / self.lam))  # [B, C]
        loss_per_sample = loss_per_elem.mean(dim=1)  # average over classes

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            return loss_per_sample

# -------------------------
# Data & stats
# -------------------------
def transform_data(dataset: pd.DataFrame, max_length=512, sampler: Optional[WeightedRandomSampler] = None,
                   shuffle: Optional[bool] = None):
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


def compute_class_counts(train_df: pd.DataFrame) -> np.ndarray:
    counts = np.zeros(_NUM_CLASSES, dtype=np.int64)
    for _, row in train_df.iterrows():
        types = ast.literal_eval(row["paraphrase_type_ids"])
        for t in types:
            if t in _TYPE_TO_IDX:
                counts[_TYPE_TO_IDX[t]] += 1
    return counts


def compute_class_priors(train_df: pd.DataFrame) -> np.ndarray:
    counts = compute_class_counts(train_df).astype(np.float64)
    N = max(len(train_df), 1)
    p = np.clip(counts / N, 1e-6, 1 - 1e-6)
    return p  # [C]


def compute_instance_weights(df: pd.DataFrame) -> np.ndarray:
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

# -------------------------
# Threshold tuning & eval
# -------------------------
def tune_thresholds_for_mcc(model: nn.Module, dev_loader: DataLoader, device,
                            num_classes: int = _NUM_CLASSES,
                            grid: Optional[Sequence[float]] = None,
                            prior_logit: Optional[torch.Tensor] = None,
                            tau: float = 0.0):
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

    P = np.vstack(probs_list)
    Y = np.vstack(y_list).astype(int)

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

# -------------------------
# Plotting & logging helpers
# -------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _save_plot(x, y, xlabel, ylabel, title, outpath):
    plt.figure()
    plt.plot(x, y)  # single series, default style (no explicit colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def _save_metrics_csv(history: dict, out_csv: str):
    df = pd.DataFrame(history)
    df.to_csv(out_csv, index=False)

# -------------------------
# Train loop (returns history for plots)
# -------------------------
def train_model(model, train_data, dev_data, device, loss_fn, epochs=5, lr=1e-5):
    optimizer = AdamW(model.parameters(), lr=lr)

    history = {
        "epoch": [],
        "train_loss": [],
        "dev_acc_0p5": [],
        "dev_mcc_0p5": [],
    }

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
        acc, matt = evaluate_model(model, dev_data, device)  # raw view: fixed 0.5, no adj
        print(f"Epoch {epoch+1}: dev accuracy :: {acc:.3f}, dev loss :: {train_loss:.3f}, MCC :: {matt:.3f}")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["dev_acc_0p5"].append(acc)
        history["dev_mcc_0p5"].append(matt)

    return model, history

# -------------------------
# Misc
# -------------------------
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

    # DBLoss params
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--mu", type=float, default=0.9)     # 0.9 if few labels/sample; 0.05 if many labels/sample
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--kappa", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.0)  # 0.0 => DB-0FL; try 2.0 for focalized DB

    # Prior-aware logit adjustment
    p.add_argument("--logit_adjust_tau", type=float, default=0.5, help="0 disables; try 0.25–1.0")

    # Threshold tuning grid
    p.add_argument("--th_grid_start", type=float, default=0.02)
    p.add_argument("--th_grid_end", type=float, default=0.98)
    p.add_argument("--th_grid_steps", type=int, default=49)

    # Weighted sampler
    p.add_argument("--use_weighted_sampler", action="store_true")

    # Output dirs
    p.add_argument("--plot_dir", type=str, default="plots_db")
    p.add_argument("--metrics_dir", type=str, default="metrics_db")
    return p.parse_args()

# -------------------------
# Main
# -------------------------
def finetune_paraphrase_detection(args):
    seed_everything(args.seed)
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    model = BartWithClassifier(num_labels=_NUM_CLASSES, dropout=args.dropout).to(device)

    train_df = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_df  = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")

    # Split train/dev
    train_split = train_df.sample(frac=0.8, random_state=42).reset_index(drop=True)
    dev_split   = train_df.drop(train_split.index).reset_index(drop=True)

    # --- DBLoss stats from TRAIN SPLIT ONLY ---
    counts = compute_class_counts(train_split)  # [C]
    dbloss = DBLoss(
        class_counts=torch.tensor(counts, dtype=torch.float),
        alpha=args.alpha, beta=args.beta, mu=args.mu, lam=args.lam, kappa=args.kappa, gamma=args.gamma, reduction="mean"
    )
    dbloss.set_num_instances(len(train_split))

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
    print(f"DBLoss params -> alpha={args.alpha}, beta={args.beta}, mu={args.mu}, lam={args.lam}, kappa={args.kappa}, gamma={args.gamma}")
    if args.use_weighted_sampler:
        print("Using WeightedRandomSampler to upsample rare-label instances.")
    if args.logit_adjust_tau > 0:
        print(f"Using logit adjustment with tau={args.logit_adjust_tau}")

    # ---- Train (collect history) ----
    model, history = train_model(model, train_loader, dev_loader, device, loss_fn=dbloss,
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

    # For reference: fixed 0.5 with adj too
    acc_raw, mcc_raw = evaluate_model(
        model, dev_loader, device, thresholds=None,
        prior_logit=prior_logit, tau=args.logit_adjust_tau
    )
    print(f"Dev (fixed 0.5)   accuracy: {acc_raw:.3f}  MCC: {mcc_raw:.3f}")

    # ---- Save metrics + plots ----
    _ensure_dir(args.metrics_dir)
    _ensure_dir(args.plot_dir)

    # Append final tuned metrics at the last epoch row for convenience
    history["dev_acc_tuned_final"] = [None] * (len(history["epoch"]) - 1) + [acc_t]
    history["dev_mcc_tuned_final"] = [None] * (len(history["epoch"]) - 1) + [mcc_t]
    metrics_csv = os.path.join(args.metrics_dir, "dbloss_metrics.csv")
    _save_metrics_csv(history, metrics_csv)
    print(f"Saved metrics CSV to {metrics_csv}")

    # Plots (per-epoch curves)
    x = history["epoch"]
    _save_plot(x, history["train_loss"], "Epoch", "Train Loss",
               "DBLoss: Train Loss per Epoch", os.path.join(args.plot_dir, "train_loss.png"))
    _save_plot(x, history["dev_acc_0p5"], "Epoch", "Dev Accuracy (0.5)",
               "DBLoss: Dev Accuracy (fixed 0.5) per Epoch", os.path.join(args.plot_dir, "dev_acc_0p5.png"))
    _save_plot(x, history["dev_mcc_0p5"], "Epoch", "Dev MCC (0.5)",
               "DBLoss: Dev MCC (fixed 0.5) per Epoch", os.path.join(args.plot_dir, "dev_mcc_0p5.png"))
    print(f"Saved plots to {args.plot_dir}")

    # ---- Test predictions using tuned thresholds (+ optional logit adjustment) ----
    test_ids = test_df["id"]
    test_results = test_model(
        model, test_loader, test_ids, device,
        thresholds=opt_thresholds, prior_logit=prior_logit, tau=args.logit_adjust_tau
    )
    os.makedirs("predictions/bart", exist_ok=True)
    out_csv = "predictions/bart/etpc-paraphrase-detection-test-output.csv"
    test_results.to_csv(out_csv, index=False)
    print(f"Wrote predictions to {out_csv}")

if __name__ == "__main__":
    args = get_args()
    finetune_paraphrase_detection(args)
