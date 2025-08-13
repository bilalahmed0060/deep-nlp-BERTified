import argparse
import random
import ast
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
from sklearn.metrics import matthews_corrcoef
from optimizer import AdamW  # keep your local optimizer import

TQDM_DISABLE = False


# =========================
# Model (return LOGITS)
# =========================
class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=26):
        super(BartWithClassifier, self).__init__()
        self.bart = BartModel.from_pretrained("facebook/bart-large")
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)  # [B, C]
        return logits


# =========================
# Distribution-Balanced Loss
# =========================
class DBLoss(nn.Module):
    """
    Distribution-Balanced Loss (DB) for multi-label classification.

    Components:
      1) Re-balanced weighting r_hat per sample/label using inverse class frequencies.
      2) Negative-Tolerant Regularization (NTR): class-specific bias shift + lambda on negatives.
      3) Optional focal modulation with gamma (set gamma=0.0 for DB-0FL).

    Args:
        class_counts: tensor [C] of positive counts per class (from TRAIN split only).
        alpha, beta, mu: parameters for squashing r -> r_hat = alpha + sigmoid(beta * (r - mu)).
        lam (lambda): scales negative branch; loss divides negative term by lam.
        kappa: scales class-specific bias v_i; v_i = -kappa * log(p_i/(1-p_i)).
        gamma: focal focusing parameter; 0.0 disables focal part.
        reduction: "mean"|"sum"|"none"
    """
    def __init__(
        self,
        class_counts: torch.Tensor,
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

        counts = class_counts.clone().float()
        counts = torch.clamp(counts, min=1.0)  # avoid zero
        self.register_buffer("class_counts", counts)     # [C]
        self.register_buffer("inv_counts", 1.0 / counts) # [C]

        # Will be set based on #instances in training split
        self.N_instances = None
        self.bias_ready = False
        self.register_buffer("v", torch.zeros_like(counts))  # class-specific bias [C]

    def set_num_instances(self, N_instances: int):
        self.N_instances = max(int(N_instances), 1)
        p = self.class_counts / float(self.N_instances)            # p_i = n_i / N_instances
        p = torch.clamp(p, min=1e-6, max=1.0 - 1e-6)
        b_hat = torch.log(p / (1.0 - p))                           # \hat{b}_i
        v = -self.kappa * b_hat                                    # v_i
        self.v = v
        self.bias_ready = True

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not self.bias_ready:
            raise RuntimeError("Call DBLoss.set_num_instances(N) before training.")

        B, C = logits.shape
        device = logits.device

        inv_counts = self.inv_counts.to(device)      # [C]
        v = self.v.to(device).view(1, C)             # [1, C]
        y = targets

        # Re-balanced weighting (per sample/label)
        pos_mask = (y > 0).float()                   # [B, C]
        denom = torch.matmul(pos_mask, inv_counts)   # [B]
        default_denom = inv_counts.sum()
        denom = torch.where(denom > 0, denom, torch.full_like(denom, default_denom))
        denom = denom.view(B, 1)

        r = inv_counts.view(1, C) / denom            # [B, C]
        r_hat = self.alpha + torch.sigmoid(self.beta * (r - self.mu))  # [B, C]

        # Negative-tolerant regularization (shift by v, scale negatives by lam)
        z_shift = logits - v
        q_pos = torch.sigmoid(z_shift)               # positives branch
        q_neg = torch.sigmoid(self.lam * z_shift)    # negatives branch

        one_minus_q_pos = 1.0 - q_pos
        one_minus_q_neg = 1.0 - q_neg

        if self.gamma > 0.0:
            # focalized DB
            pos_term = - (one_minus_q_pos.clamp_min(self.eps) ** self.gamma) * torch.log(q_pos.clamp_min(self.eps))
            neg_term = - (q_neg.clamp_min(self.eps) ** self.gamma) * torch.log(one_minus_q_neg.clamp_min(self.eps))
        else:
            # DB-0FL
            pos_term = - torch.log(q_pos.clamp_min(self.eps))
            neg_term = - torch.log(one_minus_q_neg.clamp_min(self.eps))

        # Combine: weight by r_hat; divide negatives by lam; mean over classes
        loss_per_elem = r_hat * (y * pos_term + (1.0 - y) * (neg_term / self.lam))  # [B, C]
        loss_per_sample = loss_per_elem.mean(dim=1)  # [B]

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            return loss_per_sample


# =========================
# Data utils
# =========================
def transform_data(dataset: pd.DataFrame, max_length=512):
    """
    Turn DataFrame into DataLoader (with labels if present).
    Drops types: 12, 19, 20, 23, 27 -> total 26 labels.
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    valid_types = [i for i in range(1, 32) if i not in {12, 19, 20, 23, 27}]
    type_to_idx = {t: i for i, t in enumerate(valid_types)}

    input_ids_list = []
    attention_masks = []
    labels = []

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
            label_vec = [0] * len(valid_types)
            for t in types:
                if t in type_to_idx:
                    label_vec[type_to_idx[t]] = 1
            labels.append(label_vec)

    input_ids = torch.tensor(input_ids_list)
    attention_mask = torch.tensor(attention_masks)

    if has_labels:
        labels = torch.tensor(labels, dtype=torch.float)
        dataset = TensorDataset(input_ids, attention_mask, labels)
    else:
        dataset = TensorDataset(input_ids, attention_mask)

    return DataLoader(dataset, batch_size=32, shuffle=has_labels)


def compute_class_counts(train_df: pd.DataFrame) -> np.ndarray:
    """
    Counts positives per class across the train split (26 classes after dropping).
    """
    valid_types = [i for i in range(1, 32) if i not in {12, 19, 20, 23, 27}]
    type_to_idx = {t: i for i, t in enumerate(valid_types)}
    counts = np.zeros(len(valid_types), dtype=np.int64)
    for _, row in train_df.iterrows():
        types = ast.literal_eval(row["paraphrase_type_ids"])
        for t in types:
            if t in type_to_idx:
                counts[type_to_idx[t]] += 1
    return counts


def average_labels_per_instance(train_df: pd.DataFrame) -> float:
    lengths = []
    for _, row in train_df.iterrows():
        types = ast.literal_eval(row["paraphrase_type_ids"])
        lengths.append(len(types))
    return float(np.mean(lengths)) if lengths else 0.0


# =========================
# Train / Eval / Test
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
            logits = model(input_ids=input_ids, attention_mask=attention_mask)  # [B, C]
            loss = loss_fn(logits, labels)  # DBLoss expects logits + targets
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        train_loss /= max(1, num_batches)
        acc, matt = evaluate_model(model, dev_data, device)
        print(f"Epoch {epoch+1}: dev accuracy :: {acc:.3f}, dev loss :: {train_loss:.3f}, MCC :: {matt:.3f}")
    return model


def evaluate_model(model, test_data, device):
    """
    Computes mean per-label accuracy and mean per-label MCC.
    """
    all_pred = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            predicted_labels = (probs > 0.5).int()

            all_pred.append(predicted_labels.cpu())
            all_labels.append(labels.cpu())

    all_predictions = torch.cat(all_pred, dim=0).numpy()
    all_true_labels = torch.cat(all_labels, dim=0).numpy()

    accuracies = []
    matthews_coefficients = []
    for label_idx in range(all_true_labels.shape[1]):
        correct_predictions = np.sum(all_true_labels[:, label_idx] == all_predictions[:, label_idx])
        total_predictions = all_true_labels.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)

        matth_coef = matthews_corrcoef(all_true_labels[:, label_idx], all_predictions[:, label_idx])
        matthews_coefficients.append(matth_coef)

    accuracy = float(np.mean(accuracies))
    matthews_coefficient = float(np.mean(matthews_coefficients))
    model.train()
    return accuracy, matthews_coefficient


def test_model(model, test_data, test_ids, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask = batch[:2]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).int().cpu().numpy().tolist()
            preds.extend(predictions)
    return pd.DataFrame({"id": test_ids, "Predicted_Paraphrase_Types": preds})


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    return args


# =========================
# Main
# =========================
def finetune_paraphrase_detection(args):
    model = BartWithClassifier()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model.to(device)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")

    # Split train/dev
    train_split = train_dataset.sample(frac=0.8, random_state=42)
    dev_split = train_dataset.drop(train_split.index)

    # --- DBLoss stats from TRAIN SPLIT ONLY ---
    counts = compute_class_counts(train_split)  # np.array [C]
    avg_labels = average_labels_per_instance(train_split)
    # Heuristic for mu: high mu when few labels per sample; low mu when many
    mu = 0.9 if avg_labels <= 2.0 else 0.05

    dbloss = DBLoss(
        class_counts=torch.tensor(counts, dtype=torch.float),
        alpha=0.1, beta=10.0, mu=mu, lam=2.0, kappa=0.05, gamma=2.0, reduction="mean"
    )
    dbloss.set_num_instances(len(train_split))

    # DataLoaders
    train_data = transform_data(train_split)
    dev_data = transform_data(dev_split)
    test_data = transform_data(test_dataset)

    print(f"Loaded {len(train_dataset)} training samples.")
    print(f"Avg labels per instance (train split): {avg_labels:.2f}. Using mu={mu}.")

    # Train with DBLoss
    model = train_model(model, train_data, dev_data, device, loss_fn=dbloss, epochs=args.epochs, lr=args.lr)

    print("Training finished.")

    accuracy, matthews_corr = evaluate_model(model, dev_data, device)
    print(f"The accuracy of the model is: {accuracy:.3f}")
    print(f"Matthews Correlation Coefficient of the model is: {matthews_corr:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device)
    test_results.to_csv("predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
