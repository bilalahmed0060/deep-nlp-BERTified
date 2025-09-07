# dbloss_hparam_tuner.py
import argparse
import ast
import json
import os
import random
from typing import Optional, Sequence

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import trange
from transformers import AutoTokenizer, BartModel

# Prefer local optimizer.AdamW; fallback to torch
try:
    from optimizer import AdamW
except Exception:
    from torch.optim import AdamW

# -------------------------
# Repro
# -------------------------
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# -------------------------
# Label space (drop: 12,19,20,23,27)
# -------------------------
_VALID_TYPES = [i for i in range(1, 32) if i not in {12, 19, 20, 23, 27}]
_TYPE_TO_IDX = {t: i for i, t in enumerate(_VALID_TYPES)}
_NUM_CLASSES = len(_VALID_TYPES)

# -------------------------
# Model: BART + masked mean pooling (returns LOGITS)
# -------------------------
class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=_NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.bart = BartModel.from_pretrained("facebook/bart-large")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        out = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # [B,T,H]
        if attention_mask is None:
            pooled = last_hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            lengths = torch.clamp(mask.sum(dim=1), min=1.0)
            pooled = summed / lengths
        return self.classifier(self.dropout(pooled))  # logits [B,C]

# -------------------------
# DB Loss
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
        gamma: float = 0.0,
        reduction: str = "mean",
        eps: float = 1e-12,
    ):
        super().__init__()
        assert reduction in {"mean", "sum", "none"}
        self.alpha, self.beta, self.mu = alpha, beta, mu
        self.lam, self.kappa, self.gamma = lam, kappa, gamma
        self.reduction, self.eps = reduction, eps

        counts = class_counts.clone().float().clamp_min(1.0)
        self.register_buffer("class_counts", counts)
        self.register_buffer("inv_counts", 1.0 / counts)

        self.N_instances = None
        self.bias_ready = False
        self.register_buffer("v", torch.zeros_like(counts))

    def set_num_instances(self, N_instances: int):
        self.N_instances = max(int(N_instances), 1)
        p = (self.class_counts / float(self.N_instances)).clamp(1e-6, 1 - 1e-6)
        b_hat = torch.log(p / (1.0 - p))
        self.v = -self.kappa * b_hat
        self.bias_ready = True

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not self.bias_ready:
            raise RuntimeError("Call DBLoss.set_num_instances(N) before training.")

        B, C = logits.shape
        device = logits.device
        inv_counts = self.inv_counts.to(device)             # [C]
        v = self.v.to(device).view(1, C)
        y = targets

        # Re-balanced weighting r_hat
        pos_mask = (y > 0).float()                          # [B,C]
        denom = torch.matmul(pos_mask, inv_counts)          # [B]
        default_denom = inv_counts.sum()
        denom = torch.where(denom > 0, denom, torch.full_like(denom, default_denom)).view(B, 1)
        r = inv_counts.view(1, C) / denom
        r_hat = self.alpha + torch.sigmoid(self.beta * (r - self.mu))

        # Negative-tolerant regularization
        z_shift = logits - v
        q_pos = torch.sigmoid(z_shift)
        q_neg = torch.sigmoid(self.lam * z_shift)
        one_minus_q_pos = 1.0 - q_pos
        one_minus_q_neg = 1.0 - q_neg

        if self.gamma > 0.0:
            pos_term = - (one_minus_q_pos.clamp_min(self.eps) ** self.gamma) * torch.log(q_pos.clamp_min(self.eps))
            neg_term = - (q_neg.clamp_min(self.eps) ** self.gamma) * torch.log(one_minus_q_neg.clamp_min(self.eps))
        else:
            pos_term = - torch.log(q_pos.clamp_min(self.eps))
            neg_term = - torch.log(one_minus_q_neg.clamp_min(self.eps))

        loss = r_hat * (y * pos_term + (1.0 - y) * (neg_term / self.lam))  # [B,C]
        loss = loss.mean(dim=1)  # per-sample
        return loss.mean() if self.reduction == "mean" else loss.sum()

# -------------------------
# Data & helpers
# -------------------------
def transform_data(
    dataset: pd.DataFrame,
    max_length=512,
    sampler: Optional[WeightedRandomSampler] = None,
    shuffle: Optional[bool] = None,
    batch_size: int = 32,
):
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
        return DataLoader(tensor_ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(tensor_ds, batch_size=batch_size, shuffle=(has_labels if shuffle is None else shuffle))

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
    return p

def compute_instance_weights(df: pd.DataFrame) -> np.ndarray:
    priors = compute_class_priors(df)
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
# Threshold tuning / Eval
# -------------------------
def tune_thresholds_for_mcc(
    model: nn.Module,
    dev_loader: DataLoader,
    device,
    num_classes: int = _NUM_CLASSES,
    grid: Optional[Sequence[float]] = None,
    prior_logit: Optional[torch.Tensor] = None,
    tau: float = 0.0,
):
    if grid is None:
        grid = np.linspace(0.02, 0.98, 49)

    model.eval()
    probs_list, y_list = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
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

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device,
    thresholds: Optional[np.ndarray] = None,
    prior_logit: Optional[torch.Tensor] = None,
    tau: float = 0.0,
):
    all_pred, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            if prior_logit is not None and tau > 0:
                logits = logits - tau * prior_logit.to(logits.device)
            probs = torch.sigmoid(logits).cpu().numpy()
            pred = (probs > 0.5).astype(int) if thresholds is None else (probs >= thresholds).astype(int)
            all_pred.append(pred)
            all_labels.append(labels.numpy())

    Yhat = np.vstack(all_pred)
    Y = np.vstack(all_labels)
    accs, mccs = [], []
    for i in range(Y.shape[1]):
        accs.append((Y[:, i] == Yhat[:, i]).mean())
        mccs.append(matthews_corrcoef(Y[:, i], Yhat[:, i]))
    return float(np.mean(accs)), float(np.mean(mccs))

# -------------------------
# Train inner loop for trials
# -------------------------
def train_for_epochs(model, train_loader, dev_loader, device, loss_fn, lr, weight_decay, epochs):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

# -------------------------
# Objective for Optuna
# -------------------------
def make_objective(args, train_df: pd.DataFrame, dev_df: pd.DataFrame, device):

    # Precompute counts once from train split for DBLoss
    class_counts_np = compute_class_counts(train_df).astype(np.float32)
    class_counts_t = torch.from_numpy(class_counts_np)

    def objective(trial: optuna.Trial):
        # ---- Search space ----
        lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)

        alpha = trial.suggest_float("alpha", 0.05, 0.2)
        beta = trial.suggest_float("beta", 5.0, 15.0)
        mu = trial.suggest_categorical("mu", [0.05, 0.5, 0.9])
        lam = trial.suggest_float("lam", 1.5, 3.0)
        kappa = trial.suggest_float("kappa", 0.025, 0.075)
        gamma = trial.suggest_categorical("gamma", [0.0, 1.0, 2.0])

        tau = trial.suggest_float("tau", 0.0, 1.0)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        th_steps = trial.suggest_categorical("th_grid_steps", [31, 49, 61])
        use_ws = trial.suggest_categorical("use_weighted_sampler", [False, True])

        # ---- DataLoaders ----
        sampler = None
        if use_ws:
            inst_weights = compute_instance_weights(train_df)
            sampler = WeightedRandomSampler(
                weights=torch.tensor(inst_weights, dtype=torch.double),
                num_samples=len(inst_weights),
                replacement=True,
            )
        train_loader = transform_data(train_df, sampler=sampler, shuffle=(sampler is None), batch_size=batch_size)
        dev_loader   = transform_data(dev_df, shuffle=False, batch_size=batch_size)

        # ---- Model & Loss ----
        model = BartWithClassifier(num_labels=_NUM_CLASSES, dropout=dropout).to(device)
        dbloss = DBLoss(
            class_counts=class_counts_t.to(device),
            alpha=alpha, beta=beta, mu=mu, lam=lam, kappa=kappa, gamma=gamma, reduction="mean"
        )
        dbloss.set_num_instances(len(train_df))

        # ---- Short train ----
        train_for_epochs(model, train_loader, dev_loader, device, dbloss, lr, weight_decay, args.epochs_per_trial)

        # ---- Prior-aware logit adj ----
        prior_logit = None
        if tau > 0.0:
            priors = compute_class_priors(train_df)
            prior_logit_np = np.log(priors / (1.0 - priors))
            prior_logit = torch.from_numpy(prior_logit_np).float()

        # ---- Threshold tuning (optimize MCC) ----
        grid = np.linspace(0.02, 0.98, int(th_steps))
        thresholds = tune_thresholds_for_mcc(
            model, dev_loader, device, num_classes=_NUM_CLASSES,
            grid=grid, prior_logit=prior_logit, tau=tau
        )

        # ---- Dev score (objective = tuned MCC) ----
        acc_t, mcc_t = evaluate_model(
            model, dev_loader, device, thresholds=thresholds,
            prior_logit=prior_logit, tau=tau
        )

        # Intermediate report (raw 0.5 MCC) for pruning signal
        acc_raw, mcc_raw = evaluate_model(
            model, dev_loader, device, thresholds=None,
            prior_logit=prior_logit, tau=tau
        )
        trial.report(mcc_raw, step=0)
        if not args.prune_disable and trial.should_prune():
            raise optuna.TrialPruned()

        # Attach handy attrs
        trial.set_user_attr("acc_tuned", acc_t)
        trial.set_user_attr("mcc_raw", mcc_raw)
        trial.set_user_attr("acc_raw", acc_raw)

        return mcc_t

    return objective

# -------------------------
# Train best config longer & export artifacts/predictions
# -------------------------
def train_best_and_export(args, best_params, train_df, dev_df, test_df, device):
    # Loader choices
    use_ws = best_params["use_weighted_sampler"]
    sampler = None
    if use_ws:
        inst_weights = compute_instance_weights(train_df)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(inst_weights, dtype=torch.double),
            num_samples=len(inst_weights),
            replacement=True,
        )
    train_loader = transform_data(train_df, sampler=sampler, shuffle=(sampler is None), batch_size=best_params["batch_size"])
    dev_loader   = transform_data(dev_df, shuffle=False, batch_size=best_params["batch_size"])
    test_loader  = transform_data(test_df, shuffle=False, batch_size=best_params["batch_size"])

    # Model & DBLoss (best)
    class_counts_t = torch.from_numpy(compute_class_counts(train_df).astype(np.float32))
    model = BartWithClassifier(num_labels=_NUM_CLASSES, dropout=best_params["dropout"]).to(device)
    dbloss = DBLoss(
        class_counts=class_counts_t.to(device),
        alpha=best_params["alpha"],
        beta=best_params["beta"],
        mu=best_params["mu"],
        lam=best_params["lam"],
        kappa=best_params["kappa"],
        gamma=best_params["gamma"],
        reduction="mean",
    )
    dbloss.set_num_instances(len(train_df))

    optimizer = AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    for _ in trange(args.final_epochs, desc="Final train"):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = dbloss(logits, labels)
            loss.backward()
            optimizer.step()

    # Prior-aware adj
    tau = best_params["tau"]
    prior_logit = None
    if tau > 0.0:
        priors = compute_class_priors(train_df)
        prior_logit_np = np.log(priors / (1.0 - priors))
        prior_logit = torch.from_numpy(prior_logit_np).float()

    # Tune thresholds with best grid
    grid = np.linspace(0.02, 0.98, int(best_params["th_grid_steps"]))
    thresholds = tune_thresholds_for_mcc(
        model, dev_loader, device, num_classes=_NUM_CLASSES,
        grid=grid, prior_logit=prior_logit, tau=tau
    )

    # Save artifacts
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "best_dbloss_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    np.save(os.path.join(args.out_dir, "best_thresholds.npy"), thresholds)

    # Final dev metrics
    acc_t, mcc_t = evaluate_model(model, dev_loader, device, thresholds=thresholds, prior_logit=prior_logit, tau=tau)
    print(f"[FINAL DEV] tuned accuracy={acc_t:.3f}  tuned MCC={mcc_t:.3f}")

    # Test predictions
    test_ids = test_df["id"]
    preds_df = test_model(model, test_loader, test_ids, device, thresholds=thresholds, prior_logit=prior_logit, tau=tau)
    pred_path = os.path.join(args.out_dir, "etpc-paraphrase-detection-test-output.csv")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    preds_df.to_csv(pred_path, index=False)
    print(f"[FINAL TEST] wrote predictions to {pred_path}")

def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    test_ids,
    device,
    thresholds: Optional[np.ndarray] = None,
    prior_logit: Optional[torch.Tensor] = None,
    tau: float = 0.0,
):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask = batch[:2]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            if prior_logit is not None and tau > 0:
                logits = logits - tau * prior_logit.to(logits.device)
            probs = torch.sigmoid(logits).cpu().numpy()
            pred = (probs > 0.5).astype(int) if thresholds is None else (probs >= thresholds).astype(int)
            preds.extend(pred.tolist())
    return pd.DataFrame({"id": test_ids, "Predicted_Paraphrase_Types": preds})

# -------------------------
# CLI
# -------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=11711)
    p.add_argument("--use_gpu", action="store_true")

    # Tuning
    p.add_argument("--n_trials", type=int, default=40)
    p.add_argument("--epochs_per_trial", type=int, default=3)
    p.add_argument("--prune_disable", action="store_true")

    # Final train with best params
    p.add_argument("--final_epochs", type=int, default=5)

    # Output
    p.add_argument("--study_csv", type=str, default="optuna_dbloss_study.csv")
    p.add_argument("--out_dir", type=str, default="outputs_dbloss_tuned")
    return p.parse_args()

# -------------------------
# Main
# -------------------------
def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")

    # Load & split once for fairness across trials
    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_df      = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")
    train_split = train_dataset.sample(frac=0.8, random_state=42).reset_index(drop=True)
    dev_split   = train_dataset.drop(train_split.index).reset_index(drop=True)

    objective = make_objective(args, train_split, dev_split, device)

    pruner = optuna.pruners.NopPruner() if args.prune_disable else optuna.pruners.MedianPruner(n_startup_trials=8)
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="DBLoss-BART-MCC")
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    # Save all trials
    df = study.trials_dataframe()
    df.to_csv(args.study_csv, index=False)
    print(f"Saved study trials to {args.study_csv}")

    # Normalize best params and types
    best_params = study.best_trial.params
    cast = {
        "alpha": float, "beta": float, "mu": float, "lam": float, "kappa": float, "gamma": float,
        "lr": float, "weight_decay": float, "dropout": float, "tau": float,
        "batch_size": int, "th_grid_steps": int, "use_weighted_sampler": bool
    }
    for k, fn in cast.items():
        best_params[k] = fn(best_params[k])
    print("Best tuned params:", json.dumps(best_params, indent=2))

    # Train longer with best & export thresholds/preds
    train_best_and_export(args, best_params, train_split, dev_split, test_df, device)

if __name__ == "__main__":
    main()
