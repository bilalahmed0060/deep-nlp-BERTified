import argparse
import os
from pprint import pformat
import random
import re
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd 
from bert import BertModel
from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW
from torch.optim.lr_scheduler import LambdaLR 
from torch.cuda.amp import autocast, GradScaler
import math
from transformers import BertTokenizerFast
from nltk.stem import PorterStemmer, SnowballStemmer
TQDM_DISABLE = True

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
def rdrop_bce_loss(logits1, logits2, labels, alpha=2.0):
    """
    R-Drop for binary logit.
    loss = mean(BCE(logits1,y), BCE(logits2,y)) + alpha * SymmetricKL(sigmoid(logits1) || sigmoid(logits2))
    """
    bce1 = F.binary_cross_entropy_with_logits(logits1, labels, reduction="mean")
    bce2 = F.binary_cross_entropy_with_logits(logits2, labels, reduction="mean")
    # Bernoulli parameters
    p1 = torch.sigmoid(logits1).clamp(1e-7, 1 - 1e-7)
    p2 = torch.sigmoid(logits2).clamp(1e-7, 1 - 1e-7)
    # Symmetric KL between Bernoulli(p1) and Bernoulli(p2)
    kl12 = p1 * (p1 / p2).log() + (1 - p1) * ((1 - p1) / (1 - p2)).log()
    kl21 = p2 * (p2 / p1).log() + (1 - p2) * ((1 - p2) / (1 - p1)).log()
    skl = (kl12 + kl21).mean()
    return 0.5 * (bce1 + bce2) + alpha * skl
class MultitaskBERT(nn.Module):
    """
    This module should use BERT for these tasks:
    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    (- Paraphrase type detection (predict_paraphrase_types))
    """
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # Select which variant to run (e.g., "mnrl", "optimized_pipeline", "mnrl_with_optimized_pipeline",
        # "sbert", "stemming"). This flag controls which heads/losses we build and how we route forward().
        self.approach = getattr(config, "approach", "mnrl")
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True
        ### TODO
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.approach == "sbert":
            self.normalize_embeddings = getattr(config, "normalize_embeddings", True)
            self.use_sbert_pooling = getattr(config, "use_sbert_pooling", True)
            self.use_pair_features = getattr(config, "use_pair_features", True) 
        
        if self.approach == "stemming" or self.approach == "mnrl" or self.approach == "optimized_pipeline" or self.approach == "mnrl_with_optimized_pipeline":
            self.paraphrase_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        elif self.approach == "sbert":
            pair_dim = (4 if self.use_pair_features else 2) * BERT_HIDDEN_SIZE
            self.paraphrase_classifier = nn.Linear(pair_dim, 1)
        if self.approach == "mnrl":
            self.proj_dim = getattr(config, "proj_dim", BERT_HIDDEN_SIZE)
            if self.proj_dim and self.proj_dim != BERT_HIDDEN_SIZE:
                self.qqp_proj = nn.Linear(BERT_HIDDEN_SIZE, self.proj_dim)
            else:
                self.qqp_proj = nn.Identity()
            self.qqp_proj_act = nn.GELU()
            # Stronger QQP classifier head using interaction features
            emb_dim = self.proj_dim if self.proj_dim else BERT_HIDDEN_SIZE
            # First projection: 4*emb_dim -> emb_dim
            self.qqp_hidden = nn.Linear(emb_dim * 4, emb_dim)
            self.qqp_act = nn.GELU()
            self.qqp_ln = nn.LayerNorm(emb_dim, eps=1e-12)

            # Final logit for binary paraphrase classification
            self.qqp_out = nn.Linear(emb_dim, 1)

            # Temperature used by the Multiple-Negative Ranking (InfoNCE) loss (if enabled)
            self.mnr_temperature = getattr(config, "mnr_temperature", 0.05)
        
        elif self.approach == "optimized_pipeline" or self.approach == "mnrl_with_optimized_pipeline":
            pair_in = BERT_HIDDEN_SIZE * 4   # [u, v, |u-v|, u*v]
            pair_hidden = getattr(config, "qqp_hidden", 512)
            self.qqp_ffn = nn.Sequential(
                nn.Linear(pair_in, pair_hidden),
                nn.GELU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.LayerNorm(pair_hidden),
                nn.Linear(pair_hidden, 1),
            )
        if self.approach == "mnrl_with_optimized_pipeline":
            self.proj_dim = getattr(config, "proj_dim", BERT_HIDDEN_SIZE)  # e.g., 256
            self.qqp_proj = (
                nn.Linear(BERT_HIDDEN_SIZE, self.proj_dim)
                if self.proj_dim != BERT_HIDDEN_SIZE else nn.Identity()
            )
            self.qqp_proj_act = nn.GELU()
    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them."""
        ### TODO: Completed
        if self.approach == "stemming" or self.approach == "mnrl" or self.approach == "optimized_pipeline" or self.approach == "mnrl_with_optimized_pipeline":
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) 
            return outputs['pooler_output']
        elif self.approach == "sbert":
            return self.sentence_embedding(input_ids, attention_mask)
    
    def _mean_pool(self, last_hidden_state, attention_mask):
        
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)          
        counts = mask.sum(dim=1).clamp(min=1e-9)               
        return summed / counts
    def _encode(self, input_ids, attention_mask, normalize=False):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out["last_hidden_state"] if isinstance(out, dict) else out[0]
        pooled = self._mean_pool(last_hidden, attention_mask)
        z = self.qqp_proj_act(self.qqp_proj(pooled))  
        return F.normalize(z, dim=-1) if normalize else z
    def qqp_embed_and_logit_mnrl(self, ids1, mask1, ids2, mask2):
        """Single encoder pass per side; returns classifier logit + normalized embeddings for MNR."""
        z1 = self._encode(ids1, mask1, normalize=False)  
        z2 = self._encode(ids2, mask2, normalize=False)  
        feat = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=-1)  
        feat = self.dropout(self.qqp_act(self.qqp_ln(self.qqp_hidden(feat))))
        logit = self.qqp_out(feat).squeeze(-1)  
        z1n = F.normalize(z1, dim=-1)
        z2n = F.normalize(z2, dim=-1)
        return logit, z1n, z2n
    def get_projected(self, input_ids, attention_mask, normalize=True):
        """
        Returns a projected (optionally L2-normalized) sentence embedding for MNR.
        Uses the same pooled vector as your classifier head.
        """
        e = self.forward(input_ids, attention_mask)      
        z = self.qqp_proj_act(self.qqp_proj(e))         
        return F.normalize(z, dim=-1) if normalize else z
    def qqp_embed_and_logit(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Single encoder pass per side:
        - builds the QQP classifier logit from e1,e2
        - returns L2-normalized projected embeddings for MNR
        """
        e1 = self.forward(input_ids_1, attention_mask_1)  
        e2 = self.forward(input_ids_2, attention_mask_2) 
        # classifier features
        feat = torch.cat([e1, e2, torch.abs(e1 - e2), e1 * e2], dim=-1) 
        logit = self.qqp_ffn(feat).squeeze(-1)  # [B]
        # projected embeddings for contrastive loss
        z1 = self.qqp_proj_act(self.qqp_proj(e1))
        z2 = self.qqp_proj_act(self.qqp_proj(e2))
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return logit, z1, z2
    def _pair_features(self, emb1, emb2):
        if self.use_pair_features:
            diff = torch.abs(emb1 - emb2)
            prod = emb1 * emb2
            feat = torch.cat([emb1, emb2, diff, prod], dim=-1)
        else:
            feat = torch.cat([emb1, emb2], dim=-1)
        return self.dropout(feat)
    def sentence_embedding(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_sbert_pooling:
            
            if isinstance(out, dict):
                token_embs = out.get("last_hidden_state", out.get("sequence_output", None))
                if token_embs is None:
                    token_embs = out["hidden_states"][-1] 
            else:
                token_embs = out[0]  
            emb = self._mean_pool(token_embs, attention_mask)
        else:
            # default CLS pooler
            emb = out["pooler_output"] if isinstance(out, dict) else out[1]
        if self.normalize_embeddings:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb
    
    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        if self.approach == "mnrl":
            logit, _, _ = self.qqp_embed_and_logit_mnrl(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            return logit
        elif self.approach == "optimized_pipeline" or self.approach == "mnrl_with_optimized_pipeline":
            # Encode both sentences
            e1 = self.forward(input_ids_1, attention_mask_1) 
            e2 = self.forward(input_ids_2, attention_mask_2) 
            # Interaction features
            feat = torch.cat([e1, e2, torch.abs(e1 - e2), e1 * e2], dim=-1) 
            logit = self.qqp_ffn(feat).squeeze(-1)  
            return logit
        elif self.approach == "sbert":
            emb1 = self.sentence_embedding(input_ids_1, attention_mask_1)
            emb2 = self.sentence_embedding(input_ids_2, attention_mask_2)
            feat = self._pair_features(emb1, emb2)
            logit = self.paraphrase_classifier(feat)
            return logit.squeeze(-1)
        elif self.approach == "stemming":
            emb1 = self.forward(input_ids_1, attention_mask_1)
            emb2 = self.forward(input_ids_2, attention_mask_2)
            combined = torch.cat((emb1, emb2), dim=-1)
            combined = self.dropout(combined)
            logits = self.paraphrase_classifier(combined)
            return logits.squeeze(-1)
    
   
def save_model(model, optimizer, args, config, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "model_config": config,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }
    torch.save(save_info, filepath)
    print(f"Saving the model to {filepath}.")
def multiple_negative_ranking_loss(z1, z2, temperature=0.05, symmetric=True):
    """
    z1, z2: [B, D], L2-normalized embeddings.
    Positives are (z1[i], z2[i]); all z1[i] with z2[j] (j!=i) are negatives (in-batch).
    This is the MultipleNegativesRankingLoss / InfoNCE variant.
    """
    logits = (z1 @ z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss12 = F.cross_entropy(logits, labels)
    if not symmetric:
        return loss12
    loss21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss12 + loss21)
@torch.no_grad()
def eval_qqp_simple(dataloader, model, device, threshold=0.5):
    model.eval()
    total = 0
    tp = fp = fn = tn = 0
    for batch in dataloader:
        b_ids_1 = batch["token_ids_1"].to(device)
        b_mask_1 = batch["attention_mask_1"].to(device)
        b_ids_2 = batch["token_ids_2"].to(device)
        b_mask_2 = batch["attention_mask_2"].to(device)
        labels = batch["labels"].to(device).float()
        logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        total += labels.numel()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    acc       = (tp + tn) / max(1, total)
    return acc, precision, recall, f1
# TODO Currently only trains on SST dataset!
def train_multitask(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Load data
    # Create the data and its corresponding datasets and dataloader:
    sst_train_data, _, quora_train_data, sts_train_data, etpc_train_data = load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, args.etpc_train, split="train"
    )
    sst_dev_data, _, quora_dev_data, sts_dev_data, etpc_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="train"
    )
    sst_train_dataloader = None
    sst_dev_dataloader = None
    quora_train_dataloader = None
    quora_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    etpc_train_dataloader = None
    etpc_dev_dataloader = None
    
    # QQP dataset
    if args.task == "qqp" or args.task == "multitask":
        quora_train_data = SentencePairDataset(quora_train_data, args)
        quora_dev_data = SentencePairDataset(quora_dev_data, args)
        quora_train_dataloader = DataLoader(
            quora_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=quora_train_data.collate_fn,
        )
        quora_dev_dataloader = DataLoader(
            quora_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=quora_dev_data.collate_fn,
        )
    
    config = {
       "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
        "mnr_temperature": args.mnr_temperature,
        "proj_dim": args.proj_dim,
        "grad_ckpt": args.grad_ckpt,   # <-- added
        "qqp_hidden": args.qqp_hidden,
        "use_sbert_pooling": True,
        "normalize_embeddings": True,
        "use_pair_features": True,
        "approach": args.approach,
    }
    config = SimpleNamespace(**config)
    if args.approach == "stemming":
        # helpers for stemming augmentation
        tokenizer_fast = BertTokenizerFast.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )
        stemmer = None
        if getattr(args, "stem_aug", False):
            stemmer = PorterStemmer() if args.stemmer == "porter" else SnowballStemmer("english")
    separator = "-" * 30
    print(separator)
    print("    BERT Model Configuration")
    print(separator)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print(separator)
    model = MultitaskBERT(config)
    model = model.to(device)
    lr = args.lr
    if args.approach == "mnrl" or args.approach == "sbert" or args.approach == "stemming":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif args.approach == "optimized_pipeline" or args.approach == "mnrl_with_optimized_pipeline":
        try:
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=getattr(args, "weight_decay", 0.01))
        except TypeError:
            optimizer = AdamW(model.parameters(), lr=lr)  
        steps_per_epoch = len(quora_train_dataloader) if quora_train_dataloader is not None else 0
        num_training_steps = args.epochs * steps_per_epoch
        warmup_steps = int(getattr(args, "warmup_ratio", 0.1) * num_training_steps)
        
        def lr_lambda(current_step):
            if num_training_steps == 0:
                return 1.0
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
            )
        scheduler = LambdaLR(optimizer, lr_lambda) if num_training_steps > 0 else None
        if args.approach == "mnrl_with_optimized_pipeline":
            use_amp = bool(args.use_gpu) 
            def _backward_and_step(loss):
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if getattr(args, "max_grad_norm", 1.0) and args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if getattr(args, "max_grad_norm", 1.0) and args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            scaler = GradScaler(enabled=use_amp)
    best_dev_acc = float("-inf")
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        
        if args.approach == "mnrl":
            use_amp = bool(getattr(args, "fp16", False) and args.use_gpu)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        # ...
        if args.task == "qqp" or args.task == "multitask":
            for batch in tqdm(
                quora_train_dataloader, desc=f"train-qqp-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                b_ids_1 = batch["token_ids_1"].to(device, non_blocking=True)
                b_mask_1 = batch["attention_mask_1"].to(device, non_blocking=True)
                b_ids_2 = batch["token_ids_2"].to(device, non_blocking=True)
                b_mask_2 = batch["attention_mask_2"].to(device, non_blocking=True)
                b_labels = batch["labels"].to(device, non_blocking=True).float()
                if args.approach == "mnrl":
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.zero_grad()
                if args.approach == "mnrl":
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        # ONE encoder pass per side → logit + normalized embeddings
                        logits, z1n, z2n = model.qqp_embed_and_logit_mnrl(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                        clf_loss = F.binary_cross_entropy_with_logits(logits, b_labels)
                        # in-batch negatives (B x B sim matrix is small unless batch is huge)
                        mnr_loss = multiple_negative_ranking_loss(
                            z1n, z2n,
                            temperature=model.mnr_temperature,
                            symmetric=getattr(args, "mnr_symmetric", True)
                        )
                        loss = clf_loss + args.mnr_weight * mnr_loss
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                elif args.approach == "optimized_pipeline":
                    logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                    if getattr(args, "label_smoothing", 0.0) > 0.0:
                        eps = args.label_smoothing
                        targets = b_labels * (1.0 - eps) + 0.5 * eps
                    else:
                        targets = b_labels
                    pos_w = getattr(args, "pos_weight", 0.0)
                    if pos_w and pos_w > 0.0:
                        pw = torch.tensor(pos_w, device=targets.device)
                        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
                    else:
                        loss = F.binary_cross_entropy_with_logits(logits, targets)
                    loss.backward()
                    # gradient clipping 
                    if getattr(args, "max_grad_norm", 1.0) and args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                elif args.approach == "mnrl_with_optimized_pipeline":
                    with autocast(enabled=use_amp):
                        logits, z1, z2 = model.qqp_embed_and_logit(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                        if getattr(args, "label_smoothing", 0.0) > 0.0:
                            eps = args.label_smoothing
                            targets = b_labels * (1.0 - eps) + 0.5 * eps
                        else:
                            targets = b_labels
                        # 1) BCE (with optional pos_weight)
                        pos_w = getattr(args, "pos_weight", 0.0)
                        if pos_w and pos_w > 0.0:
                            pw = torch.tensor(pos_w, device=targets.device)
                            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
                        else:
                            bce = F.binary_cross_entropy_with_logits(logits, targets)
                        # 2) MNR contrastive loss (z1,z2 already normalized)
                        mnr = multiple_negative_ranking_loss(
                            z1, z2,
                            temperature=args.mnr_temperature,
                            symmetric=args.mnr_symmetric
                        )
                        # 3) final objective
                        loss = bce + args.mnr_weight * mnr
                    # backward/step with AMP
                    _backward_and_step(loss)
                elif args.approach == "sbert":
                    logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                    loss = F.binary_cross_entropy_with_logits(logits, b_labels)
                    loss.backward()
                    optimizer.step()
                elif args.approach == "stemming":
                    logits_orig = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                    if getattr(args, "stem_aug", False):
                        # 2) Decode -> stem -> re-encode to same max length
                        max_len = b_ids_1.size(1)
                        texts1 = tokenizer_fast.batch_decode(b_ids_1, skip_special_tokens=True)
                        texts2 = tokenizer_fast.batch_decode(b_ids_2, skip_special_tokens=True)
                        def stem_line(s):
                            return " ".join(stemmer.stem(w) for w in s.split())
                        texts1_s = [stem_line(t) for t in texts1]
                        texts2_s = [stem_line(t) for t in texts2]
                        enc1 = tokenizer_fast(texts1_s, padding="max_length", truncation=True,
                                            max_length=max_len, return_tensors="pt")
                        enc2 = tokenizer_fast(texts2_s, padding="max_length", truncation=True,
                                            max_length=max_len, return_tensors="pt")
                        s_ids_1 = enc1["input_ids"].to(device)
                        s_mask_1 = enc1["attention_mask"].to(device)
                        s_ids_2 = enc2["input_ids"].to(device)
                        s_mask_2 = enc2["attention_mask"].to(device)
                        # 3) Stemmed forward
                        logits_stem = model.predict_paraphrase(s_ids_1, s_mask_1, s_ids_2, s_mask_2)
                        # 4) Consistency loss: treat stemmed as another "view"
                        loss = rdrop_bce_loss(logits_orig, logits_stem, b_labels, alpha=args.stem_alpha)
                    else:
                        # plain BCE baseline
                        loss = F.binary_cross_entropy_with_logits(logits_orig, b_labels)
                    loss.backward()
                    optimizer.step()
                train_loss += loss.item()
                num_batches += 1
        
        train_loss = train_loss / num_batches
        quora_train_acc, _, _, sst_train_acc, _, _, sts_train_corr, _, _, etpc_train_acc, _, _ = (
            model_eval_multitask(
                sst_train_dataloader,
                quora_train_dataloader,
                sts_train_dataloader,
                etpc_train_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )
        quora_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_corr, _, _, etpc_dev_acc, _, _ = (
            model_eval_multitask(
                sst_dev_dataloader,
                quora_dev_dataloader,
                sts_dev_dataloader,
                etpc_dev_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )
        train_acc, dev_acc = {
            "sst": (sst_train_acc, sst_dev_acc),
            "sts": (sts_train_corr, sts_dev_corr),
            "qqp": (quora_train_acc, quora_dev_acc),
            "etpc": (etpc_train_acc, etpc_dev_acc),
            "multitask": (
                np.mean([
                    quora_train_acc or 0,
                    sst_train_acc or 0,
                    sts_train_corr or 0,
                    etpc_train_acc or 0,
                ]),
                np.mean([
                    quora_dev_acc or 0,
                    sst_dev_acc or 0,
                    sts_dev_corr or 0,
                    etpc_dev_acc or 0,
                ]),
            ),  # TODO
        }[args.task]
        print(
            f"Epoch {epoch+1:02} ({args.task}): train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )
        if args.task == "qqp" and quora_dev_dataloader is not None:
            acc_ex, prec_ex, rec_ex, f1_ex = eval_qqp_simple(quora_dev_dataloader, model, device, threshold=0.5)
            print(f"   [QQP extra] dev: acc={acc_ex:.4f}  P={prec_ex:.4f}  R={rec_ex:.4f}  F1={f1_ex:.4f}")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            os.makedirs(os.path.dirname(args.filepath), exist_ok=True)
            save_model(model, optimizer, args, config, args.filepath)
def test_model(args):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        saved = torch.load(args.filepath)
        config = saved["model_config"]
        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")
        return test_model_multitask(args, model, device)
def get_args():
    parser = argparse.ArgumentParser()
    # Training task
    parser.add_argument(
        "--task",
        type=str,
        help='choose between "sst","sts","qqp","etpc","multitask" to train for different tasks ',
        choices=("sst", "sts", "qqp", "etpc", "multitask"),
        default="sst",
    )
    # Model configuration
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10) #10 epochs
    parser.add_argument(
        "--option",
        type=str,
        help="pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )
    parser.add_argument("--use_gpu", action="store_true")
    args, _ = parser.parse_known_args()
    # Dataset paths
    parser.add_argument("--sst_train", type=str, default="data/sst-sentiment-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/sst-sentiment-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/sst-sentiment-test-student.csv")
    parser.add_argument("--quora_train", type=str, default="data/quora-paraphrase-train.csv")
    parser.add_argument("--quora_dev", type=str, default="data/quora-paraphrase-dev.csv")
    parser.add_argument("--quora_test", type=str, default="data/quora-paraphrase-test-student.csv")
    parser.add_argument("--sts_train", type=str, default="data/sts-similarity-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-similarity-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-similarity-test-student.csv")
    # TODO
    
    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    train_split = train_dataset.sample(frac=0.8, random_state=42)
    dev_split   = train_dataset.drop(train_split.index)
    tain_path = "data/etpc-paraphrase-train_minBert.csv"
    dev_path = "data/etpc-paraphrase-dev.csv"
    dev_split.to_csv(dev_path, index=False)
    train_split.to_csv(tain_path, index=False)
    print(f"[ETPC] held‑out dev file written to {dev_path} ({len(dev_split)} rows)")
    
    parser.add_argument("--etpc_train", type=str, default="data/etpc-paraphrase-train_minBert.csv")
    parser.add_argument("--etpc_dev", type=str, default="data/etpc-paraphrase-dev.csv")
    parser.add_argument(
        "--etpc_test", type=str, default="data/etpc-paraphrase-detection-test-student.csv"
    )
    
    parser.add_argument(
        "--sst_dev_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sst_test_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-test-output.csv"
        ),
    )
    parser.add_argument(
        "--quora_dev_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--quora_test_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-test-output.csv"
        ),
    )
    parser.add_argument(
        "--sts_dev_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sts_test_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-test-output.csv"
        ),
    )
    parser.add_argument(
        "--etpc_dev_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--etpc_test_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-test-output.csv"
        ),
    )
    # Hyperparameters
    parser.add_argument("--batch_size", help="sst: 64 can fit a 12GB GPU", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-3 if args.option == "pretrain" else 1e-5,
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="If set, load BERT/tokenizer from local cache only (no internet)."
    )
    # you reference these in config/train; add them back with defaults
    parser.add_argument("--mnr_weight", type=float, default=0.2, help="Weight for Multiple Negative Ranking loss.")
    parser.add_argument("--mnr_temperature", type=float, default=0.05, help="Softmax temperature for MNR.")
    parser.add_argument("--mnr_symmetric", action="store_true", help="Apply MNR both directions (z1->z2 and z2->z1).")
    parser.add_argument("--proj_dim", type=int, default=256, help="Projection dim before QQP head/MNR. Set 768 to disable.")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed-precision training.")
    parser.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing in BERT.")
    
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Binary label smoothing for QQP [0..0.2]")
    parser.add_argument("--pos_weight", type=float, default=0.0, help="Positive class weight for BCE; 0 disables")
    # QQP head size
    parser.add_argument("--qqp_hidden", type=int, default=512)
    # qqp stemming args
    parser.add_argument("--stem_aug", action="store_true", help="Use stemming-based consistency loss for QQP")
    parser.add_argument("--stemmer", type=str, default="porter", choices=["porter", "snowball"])
    parser.add_argument("--stem_alpha", type=float, default=1.0, help="Weight of KL in stemming consistency (R-Drop style)")
    parser.add_argument(
        "--approach",
        type=str,
        help='choose between "mnrl", "optimized_pipeline", "mnrl_with_optimized_pipeline", "sbert", "stemming" to run specific approach',
        choices=("mnrl", "optimized_pipeline", "mnrl_with_optimized_pipeline", "sbert", "stemming"),
        default="mnrl",
    )

    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = get_args()
    args.filepath = f"models/{args.approach}-{args.option}-{args.epochs}-{args.lr}-{args.task}.pt"
    seed_everything(args.seed) 
    train_multitask(args)
    test_model(args)