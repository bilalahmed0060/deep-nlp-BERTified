# asl_hparam_tuner.py
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
from tqdm import tqdm, trange
from transformers import AutoTokenizer, BartModel

# --- Optimizer: use your local optimizer.AdamW if present; else fall back
try:
    from optimizer import AdamW
except Exception:
    from torch.optim import AdamW

# -------------------------
# Repro & logging
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
# Model: BART + masked mean pooling (returns LOGITS)
# -------------------------
class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=26, dropout=0.1):
        super().__init__()
        self.bart = BartModel.from_pretrained("facebook/bart-large")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B,T,H]
        if attention_mask is None:
            pooled = last_hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            lengths = torch.clamp(mask.sum(dim=1), min=1.0)
            pooled = summed / lengths
        return self.classifier(self.dropout(pooled))  # logits [B,C]


# -------------------------
# Asymmetric Loss (ASL)
# -------------------------
class AsymmetricLossMultiLabel(nn.Module):
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        clip: float = 0.1,
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
        xs_neg = torch.sigmoid(-logits)
        if self.clip is not None and self.clip > 0:
            xs_neg = torch.clamp(xs_neg + self.clip, max=1.0)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        pt = targets * xs_pos + (1.0 - targets) * xs_neg
        one_minus_pt = 1.0 - pt
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        if self.disable_torch_grad_focal:
            focal_weight = (one_minus_pt.clamp(min=self.eps)) ** gamma.detach()
        else:
            focal_weight = (one_minus_pt.clamp(min=self.eps)) ** gamma

        loss = -loss * focal_weight
        loss = loss.mean(dim=1)  # average over classes per sample
        return loss.mean() if self.reduction == "mean" else loss.sum()


# -------------------------
# Data & helpers
# -------------------------
_VALID_TYPES = [i for i in range(1, 32) if i not in {12, 19, 20, 23, 27}]
_TYPE_TO_IDX = {t: i for i, t in enumerate(_VALID_TYPES)}
_NUM_CLASSES = len(_VALID_TYPES)  # 26
TQDM_DISABLE = False


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


def compute_class_priors(train_df: pd.DataFrame) -> np.ndarray:
    counts = np.zeros(_NUM_CLASSES, dtype=np.float64)
    N = len(train_df)
    for _, row in train_df.iterrows():
        types = ast.literal_eval(row["paraphrase_type_ids"])
        for t in types:
            if t in _TYPE_TO_IDX:
                counts[_TYPE_TO_IDX[t]] += 1.0
    p = np.clip(counts / max(N, 1), 1e-6, 1 - 1e-6)
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
# Training loop (short; used inside each trial)
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

    def objective(trial: optuna.Trial):
        # ---- Search space ----
        lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        gamma_neg = trial.suggest_categorical("gamma_neg", [1.0, 2.0, 3.0, 4.0])
        clip = trial.suggest_float("clip", 0.0, 0.2)
        tau = trial.suggest_float("tau", 0.0, 1.0)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        th_steps = trial.suggest_categorical("th_grid_steps", [31, 49, 61])
        use_ws = trial.suggest_categorical("use_weighted_sampler", [False, True])

        # ---- DataLoaders (tokenize once per trial) ----
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
        asloss = AsymmetricLossMultiLabel(
            gamma_pos=0.0,
            gamma_neg=float(gamma_neg),
            clip=float(clip),
            reduction="mean",
        )

        # ---- Train for a few epochs ----
        train_for_epochs(model, train_loader, dev_loader, device, asloss, lr, weight_decay, args.epochs_per_trial)

        # ---- Prior-aware logit adjustment (optional) ----
        prior_logit = None
        if tau > 0.0:
            priors = compute_class_priors(train_df)
            prior_logit_np = np.log(priors / (1.0 - priors))
            prior_logit = torch.from_numpy(prior_logit_np).float()

        # ---- Threshold tuning on dev (optimize MCC) ----
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

        # Report for pruning (also log the raw 0.5 MCC as intermediate)
        acc_raw, mcc_raw = evaluate_model(
            model, dev_loader, device, thresholds=None,
            prior_logit=prior_logit, tau=tau
        )
        trial.report(mcc_raw, step=0)
        if args.prune_disable is False and trial.should_prune():
            raise optuna.TrialPruned()

        # Store a few extras
        trial.set_user_attr("acc_tuned", acc_t)
        trial.set_user_attr("mcc_raw", mcc_raw)
        trial.set_user_attr("acc_raw", acc_raw)

        return mcc_t  # maximize tuned MCC

    return objective


# -------------------------
# Train best & export artifacts
# -------------------------
def train_best_and_export(args, best_params, train_df, dev_df, test_df, device):
    # Build loaders with best batch size / sampler
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

    # Model & loss (best)
    model = BartWithClassifier(num_labels=_NUM_CLASSES, dropout=best_params["dropout"]).to(device)
    asloss = AsymmetricLossMultiLabel(
        gamma_pos=0.0,
        gamma_neg=float(best_params["gamma_neg"]),
        clip=float(best_params["clip"]),
        reduction="mean",
    )

    # Train longer with best params
    epochs = args.final_epochs
    optimizer = AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    for epoch in trange(epochs, desc="Final train"):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = asloss(logits, labels)
            loss.backward()
            optimizer.step()

    # Prior-aware adj
    tau = best_params["tau"]
    prior_logit = None
    if tau > 0.0:
        priors = compute_class_priors(train_df)
        prior_logit_np = np.log(priors / (1.0 - priors))
        prior_logit = torch.from_numpy(prior_logit_np).float()

    # Tune thresholds (best grid)
    grid = np.linspace(0.02, 0.98, int(best_params["th_grid_steps"]))
    thresholds = tune_thresholds_for_mcc(
        model, dev_loader, device, num_classes=_NUM_CLASSES,
        grid=grid, prior_logit=prior_logit, tau=tau
    )

    # Save artifacts
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "best_asl_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    np.save(os.path.join(args.out_dir, "best_thresholds.npy"), thresholds)

    # Print final dev metrics
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

    # Data / batching
    p.add_argument("--batch_size_default", type=int, default=32)

    # Tuning
    p.add_argument("--n_trials", type=int, default=40)
    p.add_argument("--epochs_per_trial", type=int, default=3)
    p.add_argument("--prune_disable", action="store_true")

    # Final train with best params
    p.add_argument("--final_epochs", type=int, default=5)

    # Output
    p.add_argument("--study_csv", type=str, default="optuna_asl_study.csv")
    p.add_argument("--out_dir", type=str, default="outputs_asl_tuned")
    return p.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")

    # Load data & split once (same split across trials for fairness)
    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_df      = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")
    train_split = train_dataset.sample(frac=0.8, random_state=42).reset_index(drop=True)
    dev_split   = train_dataset.drop(train_split.index).reset_index(drop=True)

    # Build objective
    objective = make_objective(args, train_split, dev_split, device)

    # Study
    pruner = optuna.pruners.NopPruner() if args.prune_disable else optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="ASL-BART-MCC")
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    # Save study table
    df = study.trials_dataframe()
    df.to_csv(args.study_csv, index=False)
    print(f"Saved study trials to {args.study_csv}")

    # Best params
    best_params = study.best_trial.params
    # Ensure numeric types
    best_params["gamma_neg"] = float(best_params["gamma_neg"])
    best_params["clip"] = float(best_params["clip"])
    best_params["tau"] = float(best_params["tau"])
    best_params["dropout"] = float(best_params["dropout"])
    best_params["lr"] = float(best_params["lr"])
    best_params["weight_decay"] = float(best_params["weight_decay"])
    best_params["batch_size"] = int(best_params["batch_size"])
    best_params["th_grid_steps"] = int(best_params["th_grid_steps"])
    best_params["use_weighted_sampler"] = bool(best_params["use_weighted_sampler"])
    print("Best tuned params:", json.dumps(best_params, indent=2))

    # Train longer with best params & export artifacts/predictions
    train_best_and_export(args, best_params, train_split, dev_split, test_df, device)


if __name__ == "__main__":
    main()
