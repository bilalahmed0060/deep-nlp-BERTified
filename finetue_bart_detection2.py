import argparse
import random
import ast

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
# Asymmetric Loss (ASL)
# =========================
class AsymmetricLossMultiLabel(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification (Ridnik et al., 2021)
      - Decouples positive/negative focusing with (gamma_pos, gamma_neg)
      - Clips easy negatives via `clip` (aka probability margin)
      - Works directly on logits for stability

    Args:
        gamma_pos: focusing for positives (common: 0.0)
        gamma_neg: focusing for negatives (common: 2.0–4.0)
        clip: small margin to clip easy negatives (common: 0.05)
        eps: numerical stability for log()
        reduction: 'mean' | 'sum' | 'none'
        disable_torch_grad_focal: detach the focal weights (stability)
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
        """
        logits: [B, C]
        targets: [B, C] in {0,1}
        """
        # Probabilities for positive/negative branches
        xs_pos = torch.sigmoid(logits)
        xs_neg = torch.sigmoid(-logits)  # == 1 - sigmoid(logits)

        # Clip easy negatives: push very-easy negatives closer to 1 (so log(xs_neg) ~ 0)
        if self.clip is not None and self.clip > 0:
            xs_neg = torch.clamp(xs_neg + self.clip, max=1.0)

        # Basic BCE (without reduction yet)
        # loss = y * log(p) + (1 - y) * log(1 - p)
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg  # [B, C]

        # Asymmetric focusing:
        # pt = p for positives, (1-p) for negatives -> here xs_pos for y=1; xs_neg for y=0
        pt = targets * xs_pos + (1.0 - targets) * xs_neg
        one_minus_pt = 1.0 - pt

        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)

        if self.disable_torch_grad_focal:
            focal_weight = (one_minus_pt.clamp(min=self.eps)) ** gamma.detach()
        else:
            focal_weight = (one_minus_pt.clamp(min=self.eps)) ** gamma

        loss *= focal_weight

        # Sum over classes, take negative (since we built log-likelihood), reduce
        loss = -loss.mean(dim=1)  # average over C per sample
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # [B]


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
            loss = loss_fn(logits, labels)  # ASL expects logits + targets
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
    # ASL hyperparams (tweak if needed)
    parser.add_argument("--gamma_pos", type=float, default=0.0)
    parser.add_argument("--gamma_neg", type=float, default=4.0)
    parser.add_argument("--clip", type=float, default=0.05)
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

    # DataLoaders
    train_data = transform_data(train_split)
    dev_data = transform_data(dev_split)
    test_data = transform_data(test_dataset)

    print(f"Loaded {len(train_dataset)} training samples.")
    print(f"ASL params -> gamma_pos={args.gamma_pos}, gamma_neg={args.gamma_neg}, clip={args.clip}")

    # Instantiate ASL
    asloss = AsymmetricLossMultiLabel(
        gamma_pos=args.gamma_pos,
        gamma_neg=args.gamma_neg,
        clip=args.clip,
        reduction="mean",
    )

    # Train with ASL
    model = train_model(model, train_data, dev_data, device, loss_fn=asloss, epochs=args.epochs, lr=args.lr)

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
