# DNLP SS25 Final Project
-   *Group name:* The Tokenizers
    
-   *Group code:* DNLP G16

-   *Group repository:* https://github.com/bilalahmed0060/deep-nlp-BERTified
    
-   *Tutor responsible:* 	Frederik Hennecke
    
-   *Group team leader:* Muhammad Usman Khanzada
    
-   *Group members:*
    - Ahmad Raza Khawaja
    - Muhammad Usman Khanzada
    - Bilal Ahmed
    - Ali Surani
    - Areesha Jiwani
  
## Setup Instructions
### Initial Setup

1)⁠ ⁠Clone the repository:
```
⁠bash
git clone https://github.com/bilalahmed0060/deep-nlp-BERTified.git
cd deep-nlp-BERTified/
```
 ⁠
2.⁠ ⁠Run the GWDG setup script:
```
⁠ bash
./setup_gwdg.sh
conda activate dnlp
```

# Project Tasks

##  1) Quora Question Pairs (QQP)

### 1) File structure
File structure for this task:
```
 ⁠project_root/
├─ multitask_classifier_qqp.py
├─ data/
│  ├─ quora-paraphrase-train.csv
│  ├─ quora-paraphrase-test-student.csv
│  └─ quora-paraphrase-dev.csv
├─ predictions/
│  └─ bert/ 
│     ├─ quora-paraphrase-dev-output.csv
      └─ quora-paraphrase-test-output.csv
```

### 2) Execution modes
1) Baseline:

```
python multitask_classifier.py \
--task=qqp \
--option=finetune \
--use_gpu \
```
2) Improvement 1: **Multiple Negative Ranking Loss (MNRL) + Optimal Hyperparameters**
   
```
python multitask_classifier_qqp.py \
--task=qqp --option=finetune --use_gpu \
--lr=2.2e-5 --batch_size=16 --hidden_dropout_prob=0.1 \
--approach=mnrl
```

3) Improvement 2: **Optimized Pipeline (Stronger Head + Training Tweaks)**
```
python multitask_classifier_qqp.py \
--task=qqp --option=finetune --use_gpu \
--lr=2.2e-5 --label_smoothing=0.05  \
--approach=optimized_pipeline
```

4) Improvement 3: **MNRL + Optimized Pipeline + Optimal Hyperparameters**
```
python multitask_classifier_qqp.py \
--task=qqp --option=finetune --use_gpu \
--lr=3.0e-5 --batch_size=16 \
--warmup_ratio=0.06 --label_smoothing=0.05 \
--pos_weight=1.2 --mnr_temperature=0.05 --mnr_symmetric \
--proj_dim=128 --hidden_dropout_prob=0.1 \
--approach=mnrl_with_optimized_pipeline
```

5) Improvement 4: **Sentence Bert**
```
python multitask_classifier_qqp.py \
--task=qqp --option=finetune --use_gpu \
--lr=2.2e-5 --hidden_dropout_prob=0.1 \
--approach=sbert
```

6) Improvement 5: **Stemming**
```
python multitask_classifier_qqp.py \
--task=qqp --option=finetune --use_gpu \
--lr=2.2e-5 --hidden_dropout_prob=0.1 \
--stem_aug --stemmer porter -stem_alpha=0.5 \
--approach=stemming
```

### 3) CLI quick reference — QQP
```
# General
--use_gpu                             Use CUDA if available
--seed                                Global random seed (default: 11711)
--epochs                              Training epochs (default: 10)
--lr                                  AdamW learning rate (default: 1e-5)

# Model & Tokenization
--option                              BERT training mode (default: pretrain)
--batch_size                          Samples per update (default: 64)
--max_seq_length                      Maximum token length (default: 128)

# Head (QQP) & Dropout
--hidden_dropout_prob                 Dropout rate for layers (default: 0.3)
--qqp_hidden                          Hidden units in head (default: 512)

# Optimization & Scheduling
--weight_decay                        L2 regularization strength (default: 0.01)
--warmup_ratio                        LR warmup proportion (default: 0.1)
--max_grad_norm                       Gradient clipping threshold (default: 1.0)
--fp16                                Enable mixed-precision training
--grad_ckpt                           Enable gradient checkpointing in BERT

# Loss Controls & Class Imbalance
--label_smoothing                     Binary label smoothing for QQP  (default: 0.0)
--pos_weight                          Positive class weight for BCE; 0 disables (default: 0.0)

# Contrastive (MNRL) Controls
--mnr_weight                          Weight for Multiple Negative Ranking loss (default: 0.2)
--mnr_temperature                     Softmax temperature for MNR (default: 0.05)
--mnr_symmetric                       Apply MNR both directions
--proj_dim                            Projection dim before QQP head/MNR (default: 256)
    
# Stemming Consistency (R-Drop)
--stem_aug                            Use stemming-based consistency loss for QQP
--stemmer                             Stemming algorithm choice (default: porter)
--stem_alpha                          Weight of KL in stemming consistency (default: 1.0)
    
# Approach Selector
--approach                            Choose between 5 approaches for improving QQP Baseline (default: mnrl)
                                      (choices: "mnrl", "optimized_pipeline", "mnrl_with_optimized_pipeline", "sbert", "stemming")

```


# Methodology
This task targets paraphrase identification on the Quora Question Pairs (QQP) benchmark. Our goal is to improve semantic equivalence detection by extending a standard Siamese BERT baseline into a flexible research pipeline with switchable modules and training options exposed via command-line flags. Within this framework we implement and compare several upgrades an interaction-based classification head with modern optimization, SBERT-style pooling, contrastive learning (MNRL), and a stemming-based consistency regularizer individually and in combination. We evaluate on a fixed dev split using Accuracy (primary) and F1 (secondary), and analyze training dynamics to understand convergence and generalization. We have worked on 5 improvements in this documentation which are discussed below.

**1 - Multiple Negative Ranking Loss (MNRL)**
   
- **Architecture:** We turn the baseline QQP classifier into a joint contrastive + classification model. In code, qqp_embed_and_logit() encodes each question with a shared BERT, builds interaction features `[e1, e2, |e1−e2|, e1*e2]` through a small FFN head to produce the BCE logit, and also projects each sentence embedding (`self.qqp_proj + GELU`) then L2-normalizes it for the contrastive term. The InfoNCE-style Multiple Negative Ranking loss is implemented in multiple_negative_ranking_loss(), using in-batch negatives (and optional symmetric direction). Training minimizes
`total_loss = BCEWithLogits(logit, y) + λ · MNR(z1, z2; τ)` with mixed precision and AdamW.
    
- **How this extends existing ideas:** a) Baseline reuse: Siamese BERT and BCE head are kept (shared encoder, pairwise head). b) Extensions in the Baseline for the approach: (1) Contrastive shaping of the embedding space via MNR on normalized projections; (2) a stronger interaction head (hidden layer + LayerNorm + GELU) rather than a single linear layer; (3) efficient in-batch negatives so no mined negatives are required; (4) engineering supports: AMP, optional `grad-ckpt`, linear warmup/decay, CLI flags to expose all knobs.

- **Why it helps:** BCE alone only pushes the decision boundary. MNRL pulls true pairs together and pushes all other pairs apart in the global embedding geometry, so the head sees cleaner, more separable inputs, especially for lexically different but semantically identical questions. This reduces near-duplicate errors and improves generalization.

- **Evaluation protocol & metrics:** We follow the project’s eval (`model_eval_multitask` + `eval_qqp_simple`). On QQP dev we report Accuracy and F1 (with Precision/Recall). We save the best dev checkpoint. Result (QQP dev): `87.7% accuracy` (`three-digit precision: 0.877`), improving the `baseline 0.781` and edging the optimized head alone (≈0.867–0.872 across runs).

**2 - Optimized Pipeline (Stronger Head + Training Tweaks)**

- **Architecture:** We kept the BERT encoder and replaced the baseline “concat → linear” QQP head with a stronger interaction head: we form features `[e1, e2, |e1−e2|, e1*e2]` from the two sentence embeddings, pass them through a small FFN (Linear → GELU → Dropout → LayerNorm → Linear), and train with BCEWithLogits plus practical tweaks (AdamW with weight decay, linear warmup/decay LR scheduler, gradient clipping, optional label smoothing, and pos_weight for class imbalance).
     
- **How this extends existing ideas:** We build directly on the standard Siamese BERT baseline by adding classic pairwise interaction features for sentence pairs and modern optimization best practices from Transformer finetuning (AdamW, warmup/decay, clipping), without changing the backbone or its pretraining.
     
- **Why it helps:** The interaction features encode symmetry and magnitude relations that a simple concat layer misses, letting the head learn clearer boundaries between paraphrase and non-paraphrase; meanwhile, AdamW + warmup/decay and clipping stabilize updates and reduce overfitting, and label smoothing/pos_weight improve robustness on borderline or noisy labels.
     
- **Evaluation protocol & metrics:** We train on QQP for `10 epochs` and select the checkpoint with the best dev accuracy. We report Accuracy (primary) and Precision/Recall/F1 (secondary) computed from dev predictions, with three-decimal precision (e.g., `accuracy = 0.867`)

**3 - MNRL + Optimized Pipeline**
   
- **Architecture:** We combined the Optimized interaction head with a contrastive branch: sentence embeddings (BERT `[CLS]`) feed (a) an interaction FFN head using `[e1, e2, |e1−e2|, e1*e2]` (Linear → GELU → Dropout → LayerNorm → Linear) for the QQP logit, and (b) a projection to `proj_dim + GELU + L2-normalization` for a Multiple Negative Ranking (InfoNCE) loss. Training uses a joint objective `BCEWithLogits + λ·MNR`, with `temperature τ`, alongside AdamW (weight decay), linear warmup/decay, gradient clipping, and optional AMP.

- **How this extends existing ideas:** It merges two proven ideas, pairwise interaction features (stronger classifier capacity) and contrastive learning (better global embedding geometry) without changing the BERT backbone or requiring external pretraining, exposing CLI knobs (`--mnr_weight`, `--mnr_temperature`, `--proj_dim`, `--qqp_hidden`, etc.) to balance both signals.

- **Why it helps:** The interaction head sharpens local decision boundaries between paraphrase/non-paraphrase, while the MNR loss organizes the global embedding space by pulling true pairs together and pushing in-batch negatives apart; together, the classifier sees cleaner, more separable inputs, improving robustness on wording-diverse duplicates.

- **Evaluation protocol & metrics:** We train for `10 epochs` and keep the checkpoint with the best dev accuracy. We report Accuracy (primary) and Precision/Recall/F1 (secondary) to three-decimal precision (e.g., 0.872). In our run, the hybrid reached `0.872 dev accuracy` (≈87.2%), close to the best MNRL-only setting and above the baseline.

**4 - Sentence Bert**

- **Architecture:** We swapped the baseline’s `[CLS]` pooling for an SBERT-style encoder that mean-pools the last_hidden_state with the attention mask (optionally L2-normalizing), builds pairwise interaction features `[u, v, |u−v|, u*v]`, applies dropout, and uses a single linear layer to predict the QQP logit with BCEWithLogits no contrastive pretraining or deeper head.

- **How this extends existing ideas:** Starting from the standard Siamese BERT baseline (concat + linear over `[CLS]`), we adopt SBERT’s pooling to better capture sentence-level semantics and keep lightweight interaction features; this reuses the same backbone/training loop while changing only the pooling strategy and pair feature construction.

- **Why it helps:** Attention-masked mean pooling aggregates information over all tokens, producing sentence vectors that are less noisy than `[CLS]`, and interaction features add simple symmetry/magnitude cues; however, without a contrastive objective or a deeper head, it struggles to separate hard, lexically diverse paraphrases, so gains are modest.

- **Evaluation protocol & metrics:** We train on QQP for `10 epochs` and retain the best dev checkpoint by Accuracy; we report Accuracy (primary) and Precision/Recall/F1 (secondary) to three-decimal precision. In our run, SBERT reached `0.811 dev accuracy` (≈81.1%), a small lift over the baseline (0.781) but below the optimized head and MNRL variants.

**5 - Stemming**

- **Architecture:** We kept the baseline BERT setup ( `[CLS] pooling`, concat [e1; e2], single linear head, BCEWithLogits ) but added an R-Drop–style consistency loss with stemming: for each QQP pair we decode, stem both questions (Porter/Snowball), re-tokenize, run a second forward pass, and train with
`loss = ½(BCE(orig) + BCE(stem)) + α · SymmetricKL(σ(orig) || σ(stem))` (see `rdrop_bce_loss`, `--stem_aug`, `--stemmer`, `--stem_alpha`).

- **How this extends existing ideas:** It builds on the Siamese BERT baseline by adding a data/view consistency regularizer rather than changing the encoder or head—using classic text normalization (stemming) to create a second “view” and enforcing output agreement between original and stemmed inputs.

- **Why it helps:** Stemming reduces sensitivity to harmless word-form changes, so the model focuses on meaning instead of inflection; the R-Drop term penalizes disagreement between the two views, improving robustness—though over-aggressive stemming can drop meaning, which limits gains.

- **Evaluation protocol & metrics:** We train on QQP for `10 epochs` and keep the best dev checkpoint by Accuracy; we report Accuracy (primary) and Precision/Recall/F1 (secondary). This variant achieved `0.795 dev accuracy` (79.5%), a +0.014 lift over the baseline (0.781) but below SBERT (0.811), the Optimized Pipeline, and MNRL.

# Experiments
**1 - Multiple Negative Ranking Loss (MNRL)**
- **Objective**: Replace the baseline BERT model and train the QQP paraphrase classifier jointly with a contrastive objective (Multiple Negative Ranking Loss, InfoNCE-style) so that true pairs are pulled together and in-batch negatives are pushed apart, while still optimizing the usual BCE classification loss.
- **Expectations**: A clear accuracy gain over the baseline BCE-only model and a modest edge over a stronger classifier head. The MNR objective should pull true pairs together and push non-pairs apart, giving better generalization especially on hard negatives.
- **Changes wrt to Baseline**:
  
    1. Embedding: Baseline used BERT’s pooler_output (`[CLS]`); we now mean-pool tokens with an attention mask, project to 256d with `GELU`, and L2-normalize for contrastive learning.
    2. Head: Baseline concatenated `[e1; e2]` and fed a single linear layer; we now use interaction features `[z1, z2, |z1−z2|, z1*z2]` followed by a hidden layer, LayerNorm, GELU, then a final linear.
    3. Loss: Baseline optimized only `BCEWithLogits`; we now train with BCE plus a weighted Multiple Negative Ranking (InfoNCE) term using in-batch negatives with configurable temperature/weight.
    4. Training: Baseline ran standard FP32 with blocking transfers; we now enable AMP, optional gradient checkpointing, non-blocking .to() calls, and expose CLI flags (`--mnr_weight`, `--mnr_temperature`, `--mnr_symmetric`, `--proj_dim`, `--fp16`, `--grad_ckpt`, `--max_seq_length`).

- **Hyperparameter Tuning**: Across sweeps, the learning rate from `1e-5 to 5e-5` showed a clear optimal: 2.2e-5 at batch 16 delivered the best dev results, with 1e-5 close behind. Batch sizes 16–32 worked best, large enough to give MNRL more in-batch negatives and stable training while 64 didn’t outperform the well-tuned smaller batches. For the contrastive term, mnr_weight `λ ≈ 0.2` and temperature `τ ≈ 0.05` were consistently reliable; pushing `λ` too high started to fight the BCE objective, and very low/high `τ` weakened the softmax contrast. Projection dims 128–256 both performed well, with the lower dim sharpening the contrastive task without overfitting noise. Finally, `dropout ≈ 0.1`, `weight_decay = 0.01`, and `grad_clip = 1.0` served as steady stabilizers throughout. (Detail of hyperparameter tuning present in Hyperparameter Optimization section.
- **Results**: MNRL `accuracy (QQP dev) is 87.7` from the best hyperparameters which is the best of the five appproaches that were tried and a sizable `+9.6 pt` over the baseline which had an accuracy of 78.1.
  Evauluation Metrics used for MNRL are Accuracy and F1 Score

| Metric             | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Epoch 6 | Epoch 7 | Epoch 8 | Epoch 9 | Epoch 10 |
|--------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|
| Dev Accuracy (%)   | 86.3    | 86.9    | 87.2    | 87.5    | 87.2    | 87.6    | 87.3    | 87.7    | 87.2    | 87.4     |
| F1 Score | 0.823      | 0.836      | 0.838      | 0.835       | 0.838      | 0.837      | 0.836       | 0.840       | 0.836     | 0.839        |




- **Discussion**: MNRL gave the biggest lift (87.7 vs. baseline 78.1), beating the Optimized Pipeline (86.7), SBERT (81.1), and Stemming (79.5) because its contrastive loss shapes the global embedding space pulling true pairs together and pushing negatives apart so the classifier faces simpler boundaries. It especially reduced errors on near-duplicate but lexically different pairs and improved both accuracy and F1, matching our expectations. Gains were sensitive to batch size (more in-batch negatives helps), temperature (~0.05 worked well), and the MNR weight (small to moderate; too large harms BCE). We also saw more robustness to surface wording changes. Compared with a stronger head alone, MNRL helps even without extra features by improving the representation geometry that the head operates on. Overall, the best behavior came from balancing BCE with a moderate MNR weight.

**2 - Optimized Pipeline (Stronger Head + Training Tweaks)**
- **Objective**: Strengthen the QQP baseline without contrastive learning by keeping the BERT encoder fixed, but replacing the simple concat+linear classifier with a richer interaction head and steadier optimization so the model can separate paraphrases more cleanly.
- **Expectations**: We expected a large gain over the baseline but slightly below MNRL, since no global embedding shaping is applied.
- **Changes wrt to Baseline**:
  
    1. Embedding: Baseline used BERT’s pooler_output (`[CLS]`) as the sentence vector; we now keep the exact same embedding (no change).
    2. Head: Baseline did a simple `[e1; e2]` → linear classifier; we now use richer interaction features `[e1, e2, |e1−e2|, e1*e2]` followed by a small FFN (Linear → GELU → Dropout → LayerNorm → Linear) for a more expressive decision boundary.
    3. Loss: Baseline trained with plain BCEWithLogitsLoss; we now still use BCE and added optional label smoothing and pos_weight to handle borderline cases and class imbalance.
    4. Training: Baseline used AdamW with a fixed LR, no scheduler, no gradient clipping; we now use `AdamW` + weight decay, linear warmup/decay scheduler, and gradient clipping, with CLI knobs to tune them.
    5. Extras: CLI flags were exposed (`--weight_decay`, `--warmup_ratio`, `--max_grad_norm`, `--label_smoothing`, `--pos_weight`, `--qqp_hidden`) and (in our broader setup) added an extra QQP eval helper for Accuracy/F1 that didn't exist in the baseline.
 
- **Results**: `QQP dev accuracy = 86.7`, up `+8.6 pts` from the baseline 78.1. We tracked the evaluation metrics: Accuracy (primary) and F1 Score for fairness on class balance.

| Metric            | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Epoch 6 | Epoch 7 | Epoch 8 | Epoch 9 | Epoch 10 |
|:------------------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:--------:|
| Dev Accuracy (%)  |  80.6   |  84.9   |  84.9   |  86.2   |  85.8   |  86.0   |  86.5   |  86.2   |  86.7   |   86.6   |
| F1 Score          |  0.775  |  0.816  |  0.819  |  0.828  |  0.827  |  0.829  |  0.832  |  0.830  |  0.832  |  0.833   |



- **Discussion**: The optimized pipeline works because the new interaction head captures how the two sentences relate (it uses their difference and element-wise product), which a simple concat+linear can’t do. AdamW with warmup/decay and gradient clipping makes training steadier and helps avoid overfitting. It’s a bit behind MNRL since it doesn’t reshape the whole embedding space so the decision boundaries improve, but not as universally for lexically diverse duplicates. RGains were consistent across seeds, with best results around a 512-unit hidden layer and moderate dropout. Overall, it beat SBERT-pooling and stemming by a clear, steady margin, without extra pretraining or risky text normalization.

**3 - MNRL + Optimized Pipeline**
- **Objective**: Replace the baseline BERT QQP classifier with a joint objective that combines standard BCE classification with a contrastive Multiple Negative Ranking Loss (InfoNCE-style), so true pairs are pulled together and in-batch negatives are pushed apart while keeping the classifier strong and training efficient.
- **Expectations**: Achieve a clear gain over the baseline BCE-only model and a small edge over a stronger interaction head alone, with the biggest improvements on hard, lexically diverse duplicates.
- **Changes wrt to Baseline**:
  
    1. Embedding: Baseline used the BERT [`CLS`] pooled vector only; we now keep `[CLS]` but add a projection to proj_dim with GELU and L2-normalization for contrastive learning.
    2. Head: Baseline did simple `[e1; e2]` → one linear layer; we now use interaction features `[e1, e2, |e1−e2|, e1*e2]` fed to a small FFN (Linear→GELU→Dropout→LayerNorm→Linear).
    3. Loss: Baseline optimized BCEWithLogits only; we now optimize BCEWithLogits plus a weighted Multiple Negative Ranking (InfoNCE) contrastive loss with in-batch negatives (tunable temperature/weight).
    4. Training: Baseline used Adam/AdamW in FP32 with fixed LR and no scheduler/AMP/clipping; we now use AdamW+weight decay, linear warmup/decay, AMP (autocast+GradScaler), optional grad clipping and label smoothing/pos-weight—exposed via CLI flags.
 
- **Results**: `QQP dev accuracy = 87.2`, up `+9.1 pts` from the baseline 78.1 which is above using optimized pipeline alone but slightly below MNRL. We tracked the evaluation metrics: Accuracy (primary) and F1 Score for fairness on class balance.

| Metric            | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Epoch 6 | Epoch 7 | Epoch 8 | Epoch 9 | Epoch 10 |
|:------------------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:--------:|
| Dev Accuracy (%)  |  81.1   |  85.4   |  86.1   |  86.8   |  86.2   |  86.9   |  86.7   |  87.2   |  87.0   |   87.2   |
| F1 Score          |  0.786  |  0.823  |  0.832  |  0.836  |  0.832  |  0.837  |  0.836  |  0.839  |  0.838  |  0.839   |
  
- **Discussion**: This hybrid approach did better than the baseline (87.2% vs 78.1%) because the interaction head makes pairwise decisions sharper and the contrastive term organizes sentence embeddings, so near-duplicates with different words are easier to separate. It was a bit lower than the best MNRL-only run (87.6%) likely because the loss balance wasn’t perfect. The results mostly matched expectations: clear gains over baseline and roughly in line with MNRL. We saw consistent patterns: bigger batches (more in-batch negatives) help, and label smoothing/pos-weight give small boosts on borderline or imbalanced cases.

**4 - Sentence Bert**
- **Objective**: Replace the baseline’s `[CLS]` pooling with SBERT-style mean pooling over token embeddings (attention-masked) to form cleaner sentence vectors for QQP, while keeping a simple binary classifier.
- **Expectations**: A modest gain over baseline by smoothing sentence representations and making reworded duplicates easier to match, though likely below contrastive/MNRL and the stronger interaction head.
- **Changes wrt to Baseline**:

    1. Embedding: Baseline used the BERT `[CLS]` pooled vector; SBERT replaces it with attention-masked mean pooling over tokens (optionally L2-normalized) to form the sentence embedding.
    2. Head: Baseline did simple `[e1; e2]` → linear; SBERT keeps a lightweight head but feeds it richer interaction features `[u, v, |u−v|, u*v]` with dropout before the final linear layer.
    3. Loss: Baseline used BCEWithLogits for QQP; SBERT keeps the same losses with no added contrastive pretraining term.
    4. Training: Baseline trained with AdamW in FP32 and no warmup/decay; SBERT uses the same setup (no MNR, no scheduler), so the primary change is the pooling strategy.
 
- **Results**: `QQP dev accuracy = 81.1%` (primary metric), a `+3.0 pts` gain over the baseline (78.1) but below the Optimized Pipeline (≈86.7) and MNRL (≈87.6); F1 Score tracked the same trend as accuracy for sentence bert approach.

| Metric            | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Epoch 6 | Epoch 7 | Epoch 8 | Epoch 9 | Epoch 10 |
|:------------------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:--------:|
| Dev Accuracy (%)  |  78.1   |  80.1   |  80.1   |  79.9   |  81.1   |  80.0   |  80.0   |  79.4   |  80.1   |   79.9   |
| F1 Score          |  0.682  |  0.717  |  0.737  |  0.742  |  0.737  |  0.739  |  0.741  |  0.737  |  0.741  |  0.739   |
  
- **Discussion**: Mean pooling likely stabilized sentence representations and gave a small lift, but it might have improved more with contrastive shaping (e.g., MNRL) or a deeper interaction head to capture subtle phrasing differences; pretraining SBERT on NLI/STS-style data could further separate hard negatives and boost accuracy but was not experimented with due to time constraints.

**5 - Stemming**
- **Objective**: Make the QQP classifier robust to morphology by enforcing prediction consistency between each original question pair and a stemmed view `(Porter/Snowball)`, while keeping the base BERT classifier unchanged.
- **Expectations**: A small, steady bump over baseline by making the model ignore harmless word-form changes.
- **Changes wrt to Baseline**:

    1. Embedding: Baseline uses BERT’s `[CLS]` pooled vector with dropout; this approach keeps `[CLS]` exactly the same (no embedding change).
    2. Head: Baseline concatenates `[e1; e2]` and applies a single linear layer; this approach uses the same concat+linear head (no change).
    3. Loss: Baseline trains with BCEWithLogits only; this approach switches to an R-Drop objective—average BCE on (original, stemmed) logits plus α·symmetric KL between their sigmoid probabilities.
    4. Training: Baseline does one forward pass per pair; this approach enables `--stem_aug` to decode → stem `(Porter/Snowball via --stemmer)` → re-tokenize to the same max length and run a second forward per batch, leaving the optimizer/loop otherwise unchanged.
 
- **Results**: Stemming consistency achieved `79.5% QQP` dev accuracy which is `+1.4 pts` over the baseline (78.1) but `−1.6 pts` behind SBERT (81.1). This approach was evaluated using metrics: Accuracy and F1 Score.

| Metric            | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Epoch 6 | Epoch 7 | Epoch 8 | Epoch 9 | Epoch 10 |
|:------------------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:--------:|
| Dev Accuracy (%)  |  76.2   |  78.0   |  78.4   |  79.5   |  78.7   |  77.8   |  78.0   |  78.0   |  78.7   |   78.8   |
| F1 Score          |  0.618  |  0.688  |  0.697  |  0.707  |  0.715  |  0.709  |  0.716  |  0.711  |  0.716  |  0.711   |
  
  This approach increases training cost because each batch requires a second forward pass on the stemmed texts.
- **Discussion**: Consistency regularization gave a small boost because it taught the model to ignore harmless word-form changes, but heavy stemming sometimes removed meaning, which limited the gains and occasionally hurt tricky cases. Results depended a lot on the α weight (too high made predictions too “blurry”) and on which stemmer was used. Training was slower since we had to run a second pass on the stemmed text. In practice, lighter cleanup (like lemmatization) or combining this idea with contrastive pretraining would likely give more reliable improvements.


## Results 

| **Quora Question Pairs (QQP)** | **Accuracy** | **F1 Score** |
|----------------|-----------|-----------|
|Baseline |78.1%           |...            | 
|Multiple Negative Ranking Loss (MNRL)         |87.5%            |            0.841| 
|**Multiple Negative Ranking Loss (MNRL) + Optimal Hyperparameters**        |**87.7%**           |            **0.840**|
|Optimized Pipeline (Stronger Head + Training Tweaks)        |86.7%|            0.833|
|MNRL + Optimized Pipeline      |87.0|            0.839|
|MNRL + Optimized Pipeline + Optimal Hyperparameters       |87.2|         0.839|
|Sentence Bert        |81.1|            0.742|
|Stemming (R-Drop Augmentation)       |79.5|            0.716|
 

### Hyperparameter Optimization 

Hyperparameter tuning is the process of picking the best values for settings the model doesn’t learn by itself like learning rate, batch size, or hidden dropout probability to maximize validation performance. We performed hyperparameter search on the best model approach for QQP: Multiple Negative Ranking Loss (MNRL) since it was already the top performer and is very sensitive to settings. We focused on the three knobs that matter most for contrastive training:

1. `Learning rate (LR)` — stability vs. speed.
2. `Batch size` — more in-batch negatives for MNRL.
3. `Dropout` — regularization.

We kept the contrastive specifics fixed to isolate these effects: `mnr_weight = 0.20`, `temperature = 0.05`, `5 epochs`, save best dev checkpoint, and report Accuracy (primary) plus P/R/F1 for sanity.

| Set | Learning Rate | Batch Size | Dropout | Best Dev Acc | Notes |
|-----|---------------|------------|---------|--------------|-------|
| 1   | 1e-5          | 64         | 0.30    | 0.871        |Larger batch; stable training. |
| 2   | 1e-5          | 32         | 0.30    | 0.873        |Slight gain over Set 1. |
| 3   | 1e-5          | 16         | 0.30    | 0.874        |Best at LR=1e-5; smaller batch helped a bit. |
| 4   | 2.2e-5        | 16         | 0.30    | 0.875        |Clear bump from higher LR. |
| 5   | 2.2e-5        | 16         | 0.20    | 0.873        |Same LR; lowering dropout didn’t help. |
| **6**   | **2.2e-5**        | **16**         | **0.10**    | **0.876**        |**New best at same LR with lower dropout.** |
| 7   | 5e-5          | 16         | 0.30    | 0.868        |Too high LR—slight drop. |
| 8   | 5e-5          | 16         | 0.20    | 0.866        |Lower dropout didn’t help at high LR. |
| 9   | 5e-5          | 16         | 0.10    | 0.868        |Even lower dropout still below best. |

The best, most reliable configuration was `LR = 2.2e-5`, `batch = 16`, `dropout = 0.1` (Set 6), with `dev accuracy = 87.6 %` on which we later ran 10 epochs and saved it as my best model as it gave me dev accuracy of 87.7 % when Iwe ran the same settings for 10 epochs.

#### What changed when we changed each hyperparameter
1) Learning rate (LR)
 - Trend: Going from `1e-5 → 2.2e-5` improved or matched the best dev accuracy (0.876), and generalized better at 2.2e-5.
 - Too high hurts: 5e-5 consistently dropped accuracy to 0.866–0.868, suggesting the joint BCE+MNR objective became less stable at that speed.
 - Takeaway: A moderate LR (~2.2e-5) is optimal for MNRL here.
2) Batch size
 - Why bigger might help: With MNRL, each extra sample in a batch becomes a negative example, so larger batches can make the contrastive signal stronger.
 - What we actually saw: Keeping LR=1e-5 and dropout=0.3, a batch of 16 scored 0.874, 32 scored 0.873, and 64 dropped to 0.871 so the smallest batch won here.
 - Bottom line: 16–32 worked best with my optimizer/schedule; just making the batch huge didn’t automatically improve accuracy.
3) Dropout
 - At `LR=2.2e-5`, batch=16: Reducing dropout from 0.30 → 0.10 gave a small gain (0.875 → 0.876).
 - At `LR=5e-5`, batch=16: Lowering dropout 0.30 → 0.20/0.10 did not help and even dipped (0.868 → 0.866/0.868).
 - Takeaway: Dropout interacts with LR—lower (0.10) worked best at a moderate LR (2.2e-5), while higher LR (5e-5) needed stronger regularization (≈0.30)

How we picked the final MNRL config (in plain English)
- Fix the contrastive knobs: set mnr_weight `λ = 0.20` and temperature `τ = 0.05`, train 5 epochs, and keep the best dev checkpoint (`seed=11711`).
- Find a good learning rate first: with batch 16, sweep LRs; 2.2e-5 gave consistently strong dev scores and the cleanest peak (0.876).
- Check batch size next: at good LRs `(1e-5, 2.2e-5)`, batch 16–32 was best; 64 didn’t help despite more in-batch negatives.
- Tune dropout as needed: after LR, adjust regularization; dropout 0.10 proved most stable at `LR=2.2e-5` (higher LR needed more dropout).
- Why this isn’t random: each tweak follows MNRL behavior batch controls negatives, LR balances BCE vs. contrastive loss, and dropout stabilizes updates.

## Visualizations 

Five approaches were performed for the improvements of the baseline task. In these visuals, we have done a comparative study on the metrics: Dev Accuracy and Training Loss.

<img width="574" height="452" alt="image" src="https://github.com/user-attachments/assets/a6377830-0789-441a-a8b3-d06268ebad0d" />

<img width="561" height="450" alt="image" src="https://github.com/user-attachments/assets/02c56848-b076-424f-b166-602a41c8ed25" />

<img width="581" height="451" alt="image" src="https://github.com/user-attachments/assets/62cbecc4-4629-40f4-96bf-4f4a07bb02a8" />

<img width="570" height="455" alt="image" src="https://github.com/user-attachments/assets/6a910967-2319-451e-b317-709a30e1c8aa" />

<img width="573" height="448" alt="image" src="https://github.com/user-attachments/assets/3158bbb0-3f44-4620-906a-73670b797672" />


**For Dev Accuracy:**

**1. Which converges fastest?:**
The Optimized Pipeline and Hybrid (MNRL + Optimized Pipeline) jump the quickest (big gains by epoch 2–3), because the interaction head immediately gives useful pairwise features; SBERT and Stemming also rise early but hit lower ceilings; MNRL improves more gradually and peaks around epoch 6–10.

**2. Which finishes highest?:**
MNRL finishes the highest at dev accuracy 87.4%, followed by the Hybrid at 87.2%, and then Optimized Pipeline at 86.6%. SBERT and Stemming peak at 81.1% and 79.5% respectively. The contrastive loss (MNRL) forms a better global embedding space, giving the classifier easier boundaries and the best final accuracy.

**3. Does faster convergence mean better in the end?:**
Not necessarily since Optimized Pipeline converges fastest but finishes below MNRL. Quick boundary learning helps early, but without contrastive pressure the embedding geometry isn’t as clean. MNRL needs a few more epochs to fully shape the space and increase further more.

**4. Patterns to note:**
Methods that add pairwise interaction features and/or a contrastive objective not only reach higher accuracy but also train more steadily in later epochs. Pure text normalization (Stemming) is fragile; representation-level changes (SBERT) gives only small gains unless you pair it with contrastive training or a stronger classifier head.

<img width="564" height="453" alt="image" src="https://github.com/user-attachments/assets/71d17720-4aa7-4718-adc4-7da76c52af56" />
<img width="564" height="453" alt="image" src="https://github.com/user-attachments/assets/3f717cc3-840a-440f-aa79-28510b2596f3" />
<img width="564" height="455" alt="image" src="https://github.com/user-attachments/assets/fe53a0f9-eadb-49c1-9612-2d305e86a7b9" />
<img width="564" height="452" alt="image" src="https://github.com/user-attachments/assets/b4ca23bd-45cb-4ee4-a4a9-3aadfe719bcb" />
<img width="564" height="451" alt="image" src="https://github.com/user-attachments/assets/0c0fe141-cb5d-4302-9b7d-6a09b0e520f7" />



**For Training Loss:**

MNRL drops loss fast and then flattens, which matches its high dev accuracy: the contrastive term quickly organizes the embedding space and the classifier settles. Optimized Pipeline also falls steadily but a touch slower than MNRL (more parameters in the head to fit); it ends slightly below MNRL. Hybrid (MNRL + Optimized head) starts with the highest loss and reduces more slowly.
SBERT and Stemming both show smooth, strong loss reductions, yet their dev accuracy lags; which is a representation of optimization is not equal to generalization case: the training objective is easier to satisfy than the true paraphrase decision.


# References 
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf).
- Gao, T., Yao, X., & Chen, D. (2021). [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/pdf/2104.08821.pdf).
- Reimers, N., & Gurevych, I. (2019). [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf).




## 3) Sentiment Analysis (SST):
### 1) Finetuning Setup 

Make sure the shell scripts are executable:
```bash
chmod +x run_train_sst_base.sh run_train_sst_rdrop.sh run_train_sst_gradual_unfreezing.sh
```
To run different models, use the corresponding script:
- **Baseline BERT:** run_train_sst_base.sh
- **BERT with R-Drop and Regularization Strategies:** run_train_sst_rdrop.sh
- **BERT with Gradual Unfreezing:** run_train_sst_gradual_unfreezing.sh

### 2) File structure
File structure for this task:
```
 ⁠project_root/
├─ multitask_classifier.py ( baseline )
├─ sst_finetun_gradual_unfreezing.py  ( gradual unfreezing )
├─ sst_finetune_rdrop.py  ( R-Drop Regularization ) 
├─ data/
│  ├─ sst-sentiment-dev.csv
│  ├─ sst-sentiment-test-student.csv
│  └─ sst-sentiment-train.csv
├─ predictions/
│  └─ bert/ 
│     └─ sst-sentiment-dev-output.csv
      └─ sst-sentiment-test-output.csv
```

# Methodology
Each group member worked on their own on one of the different tasks:

## Sentiment Analysis (SST)

In order to improve upon the baseline, we explore 2 different approaches.

### 1. Optimization and Regularization

#### 1.1. R-Drop Regularization 
- Adopted from "R-Drop: Regulized Dropout for NN (NeurIP 2021)".
- During training, each input is passed through the model twice with different dropout masks.
- A consistency loss (bidirectional KL divergence) encourages similar predictions between the two passes.
- This reduces variance from dropout, improves robustness, and smooths decision boundaries.
- The total loss combines cross-entropy with the R-Drop regularization term.
#### 1.2. Additional Training Strategies with R-Drop
##### 1.2.1. Learning Rate Scheduler 
- Applied get_linear_schedule_with_warmup to adjust the learning rate. 
- Ensures smoother and more stable training.


### 2. Layer-Wise Fine-Tuning

#### 2.1. Gradual Unfreezing 
- All BERT encoder layers are initially frozen, and only the classification head is trained.
- After the first epoch, one BERT encoder layer is unfrozen sequentially from top to bottom (closest to
the output first).
- Newly unfrozen layers are added to the optimizer so their parameters are updated during training.
- This controlled unfreezing allows the model to slowly adapt its pretrained representations to the
sentiment classification task without catastrophic forgetting.

#### 2.2. L2 Regularization
  - Implemented through weight decay in the AdamW optimizer.
  - Helps prevent overfitting by penalizing large weights.
#### 2.3. Gradient Clipping
- Applied to prevent exploding gradients.
- Improves training stability, especially for deep networks.


# Experiments

##  Sentiment Analysis (SST)
### BERT with R-Drop and Regularization Strategies
- **Objective**: Improve BERT’s performance on sentiment analysis by applying R-Drop and additional training strategies.
- **Expectations**: Achieve better generalization, improved stability during training, and higher dev accuracy compared to the baseline model.

- **Changes**: 
  - Added R-Drop by introducing KL divergence loss between two forward passes.
  - Used a linear warmup scheduler for smoother training.
  - Used AdamW optimizer with learning rate 2e-5
  
- **Results**: 
  - Achieved the best dev accuracy of 0.552, compared to 0.527 for the baseline.
  - Training accuracy improved to 0.856, showing stable optimization.
- **Discussion**: R-Drop consistently improved performance over the baseline, with learning rate tuning providing an additional boost. The linear scheduler improved training smoothness but only added a minor gain. Overall, R-Drop was the main factor driving performance improvements on SST.

### BERT with Gradual Unfreezing
- **Objective**: Improve dev accuracy of BERT on sentiment analysis by gradually fine-tuning pretrained layers.
- **Expectations**: Gradual adaptation of pretrained BERT layers will improve generalization and dev accuracy compared to training all layers at once.

- **Changes**:
  - Initially froze all BERT layers; only the classifier trained in the first epoch.
  - Sequentially unfroze one BERT encoder layer per epoch from top to bottom.
  - Updated optimizer and scheduler to include newly trainable parameters.
  - Applied gradient clipping (using torch.nn.utils.clip_grad_norm_) to stabilize training.

- **Results**: 
  - Gradual unfreezing dev accuracy: 0.537.
  - Training accuracy increased slowly (0.579), indicating controlled adaptation.
- **Discussion**: The baseline initially achieved a dev accuracy of 0.527. By gradually unfreezing BERT layers from top to bottom, the model was able to adapt pretrained features step by step. This improved dev accuracy to 0.537. We also observed that training accuracy did not increase drastically compared to previous experiments, indicating that the model was learning effectively without overfitting. Gradual unfreezing therefore allowed better utilization of pretrained knowledge for sentiment analysis on SST.

  

## Results 
### Summary of Results
| **Stanford Sentiment Treebank (SST)** | **Dev Accuracy** |
|----------------|-----------|
|Baseline |52.7%           |
|BERT + R-Drop |  **55.2%** |
|BERT + Gradual Unfreezing          | 53.7%            |


### Discussion


####  Sentiment Analysis (SST)

This project aimed to enhance the BERT model's performance for sentiment analysis on the Stanford Sentiment Treebank (SST) dataset. We began with a baseline MiniBERT model that achieved a dev accuracy of 0.527, serving as our reference point for subsequent improvements. Our goal was to explore various techniques to boost performance, including advanced training methods, fine-tuning strategies, and hyperparameter tuning.

Initially, we tested approaches such as Multiple Negative Ranking Loss, SBERT-based sentence embeddings, and changing the optimizer to Adam. These methods did not improve performance over the baseline, showing that techniques designed for semantic similarity or ranking tasks are not always effective for sentiment classification, especially on a small dataset like SST.

The first successful improvement came from R-Drop regularization. By passing the same input twice through the model with different dropout masks and enforcing consistency between the predictions, R-Drop reduced variance and improved generalization. Adjusting the learning rate to 2e-5 further boosted dev accuracy. The linear scheduler smoothed optimization but contributed only minor gains. Overall, this approach increased dev accuracy from 0.527 to 0.552, confirming that enforcing prediction consistency and stable optimization were the main factors driving improvement.

Gradual unfreezing, another fine-tuning strategy, focused on controlled layer-wise adaptation. All BERT layers were initially frozen, and only the classifier was trained during the first epoch. Layers were then sequentially unfrozen from top to bottom over 15 epochs, allowing the model to gradually adapt pretrained features. Gradient clipping was applied to stabilize training. This approach reduced overfitting and improved dev accuracy to 0.537, comparable to R-Drop. It highlights that careful adaptation of pretrained layers can be very effective for small datasets like SST.

Overall, advanced training methods like R-Drop and careful fine-tuning through gradual unfreezing each improved the model’s performance over the baseline. R-Drop enhanced robustness and prediction consistency, while gradual unfreezing leveraged pretrained knowledge in a controlled manner. These results suggest that principled fine-tuning strategies are effective for improving sentiment classification on SST.


###  Hyperparameter Tuning and Analysis 
####  Sentiment Analysis (SST)
Hyperparameter tuning was key to improving model performance. For R-Drop, the best results were achieved with a dropout probability of 0.1, 10 epochs, a learning rate of 2e-5, and a batch size of 64. Gradient clipping and a linear scheduler helped stabilize training.

For Gradual Unfreezing, BERT layers were initially frozen and then sequentially unfrozen over 15 epochs, with a learning rate of 1e-5 and gradient clipping helping the model gradually adapt pretrained features without overfitting. We observed that a smaller dropout of 0.1 reached 0.535 dev accuracy, while 0.3 slightly improved it to 0.537. A larger batch size of 64 performed better than 16 or 32. These results show that even small changes in dropout, epochs, or batch size can noticeably affect performance, emphasizing the importance of careful hyperparameter tuning.

Hyperparameter tuning was time-consuming but essential. Small changes in parameters had a big impact on model performance, showing the importance of careful tuning when working with BERT for sentiment analysis. The optimal settings we found can be a good starting point for future experiments, though they may need adjustment for different datasets or models.

## Visualizations 

### BERT with R-Drop and Regularization Strategies

#### Train Accuracy:
<img width="640" height="480" alt="RDrop_train_accu" src="https://github.com/user-attachments/assets/44e09bee-a722-44e5-a0cd-2593b2e895bb" />

#### Dev Accuracy:
<img width="640" height="480" alt="RDrop_dev_accu" src="https://github.com/user-attachments/assets/6dbf5a28-7296-40e9-8497-545d7f1acfe0" />


#### Train loss:
<img width="640" height="480" alt="RDrop_loss" src="https://github.com/user-attachments/assets/f2c5edcf-7b0a-4bc6-a6d5-b0f6ec2cd44f" />


### BERT with Gradual Unfreezing
#### Train Accuracy:
<img width="640" height="480" alt="GU_train_acc" src="https://github.com/user-attachments/assets/61d21cfa-8523-4eff-8d2e-cb417729eccd" />

#### Dev Accuracy:
<img width="640" height="480" alt="GU_dev_accu" src="https://github.com/user-attachments/assets/fd768b7e-e49f-42c3-bf15-55f8c380410b" />

#### Train loss:

<img width="640" height="480" alt="GU_loss" src="https://github.com/user-attachments/assets/84dafab5-8670-4b6f-8b82-320f06c0936c" />

### **What the curves show:**  
- **Train loss:** Both models decrease steadily; R-Drop converges faster.  
- **Train accuracy:** R-Drop rises quickly (~97.9%), Gradual Unfreezing increases gradually (~85.8%).  
- **Dev accuracy:** R-Drop peaks earlier (~55.2%), Gradual Unfreezing rises steadily (~53.7%).  
- **Interpretation:** R-Drop gives faster gains with slight fluctuations; Gradual Unfreezing is slower but more stable, leveraging pretrained features without overfitting.  

#### References
- R-Drop: Regularized Dropout for Neural Networks NeurIPS 2021.(https://arxiv.org/pdf/2106.14448)
- ULMFiT: Universal Language Model Fine-tuning for Text Classification, Howard and Ruder, 2018. (https://aclanthology.org/P18-1031.pdf)

##  4) Bart paraphrase type detection
### 1) Install extra packages
```
pip install optuna
pip install matplotlib
```

### 2) File structure
File structure for this task:
```
 ⁠project_root/
├─ bart_detection.py ( basline )
├─ finetune_bart_detection-dbloss.py  ( dbloss optimization )
├─ finetune_bart_detection_asl.py  ( ASL optimization ) 
├─ data/
│  ├─ etpc-paraphrase-train.csv
│  ├─ etpc-paraphrase-detection-test-student.csv
│  └─ etpc-paraphrase-dev.csv
├─ predictions/
│  └─ bart/ 
      └─ etpc-paraphrase-detection-test-output.csv
```

### 3) Execution modes
1) Baseline (BCE, no imbalance tricks):
Trains BART with plain binary cross-entropy (original baseline).
```
python bart_detection_baseline.py \
  --use_gpu \
  --epochs 5 --lr 1e-5 \
  --seed 11711
```
2) Distribution-Balanced Loss (DBLoss):
Re-balanced weighting + Negative-Tolerant Regularization; optional logit adjustment and per-class threshold tuning for MCC.
```
python bart_detection_dbloss.py \
  --use_gpu \
  --epochs 5 --lr 1e-5 \
  --alpha 0.1 --beta 10.0 --mu 0.9 --lam 2.0 --kappa 0.05 --gamma 0.0 \
  --logit_adjust_tau 0.5 \
  --th_grid_start 0.02 --th_grid_end 0.98 --th_grid_steps 49 \
  --use_weighted_sampler \
  --seed 11711
```

3) Asymmetric Loss (ASL):
Decoupled focusing (γ⁺/γ⁻) + easy-negative clipping; optional logit adjustment and per-class threshold tuning.
```
python finetune_bart_detection_asl.py \
  --use_gpu \
  --epochs 5 --lr 1e-5 --dropout 0.1 \
  --gamma_pos 0.0 --gamma_neg 2.0 --clip 0.1 \
  --logit_adjust_tau 0.5 \
  --th_grid_start 0.02 --th_grid_end 0.98 --th_grid_steps 49 \
  --use_weighted_sampler \
  --seed 11711
```

4) Hyperparameter Optimization (HPO) — ASL
```
python asl_hparam_tuner.py \
  --use_gpu \
  --n_trials 40 --epochs_per_trial 3 \
  --final_epochs 5 \
  --seed 11711
```

### 4) CLI quick reference — ASL loss (paraphrase type detection)
```
# General
--use_gpu                 Use CUDA if available
--seed                    Global random seed (default: 11711)
--epochs                  Training epochs (default: 5)
--lr                      AdamW learning rate (default: 1e-5)
--dropout                 Classifier dropout (default: 0.1)

# Asymmetric Loss (ASL)
--gamma_pos               Positive focusing γ⁺ (default: 0.0)
--gamma_neg               Negative focusing γ⁻ (default: 2.0)
--clip                    Easy-negative clipping ∈ [0,1] (default: 0.1)

# Calibration & thresholds
--logit_adjust_tau        Prior-aware logit adjustment τ (0 disables; e.g., 0.25–1.0; default: 0.0)
--th_grid_start           Threshold sweep start (default: 0.02)
--th_grid_end             Threshold sweep end   (default: 0.98)
--th_grid_steps           Threshold sweep steps (default: 49)

# Sampling (imbalance handling)
--use_weighted_sampler    Upsample rare-label instances via WeightedRandomSampler

# I/O (if enabled in your script)
--plot_dir                Directory for plots (e.g., plots_asl)
--metrics_dir             Directory for metrics CSV (e.g., metrics_asl)

# (Optional) Tokenization / seq length
--max_length              Tokenizer max length (default: 512)

# --- Hyperparameter Optimization (ASL tuner) ---
# Run with `asl_hparam_tuner.py`
--n_trials                Number of Optuna trials (e.g., 24–60)
--epochs_per_trial        Train epochs per trial (budget; e.g., 1–3)
--final_epochs            Full train epochs for best params (default: 5)
--prune_disable           Disable Optuna pruner (enable for debugging)
```

### Methodology
#### 1) Task & Data
- Problem: Given a sentence pair (s₁, s₂), predict 26 paraphrase types (multi-label; a sample can have multiple types).
- Labels: We map original type IDs 1–31 to a 26-class space by dropping {12, 19, 20, 23, 27} as specified.
- Metric: Primary metric is mean per-class Matthews Correlation Coefficient (MCC), averaged over the 26 labels. MCC is robust to skew and reflects both precision and recall.

#### 2) Baseline
- Model: facebook/bart-large encoder with a masked-mean pooled sentence representation (we do not rely on the first token) followed by a linear classifier (26 logits).
- Tokenization: BART tokenizer on the concatenated pair (sentence1, sentence2) with max length 512 and padding/truncation.
- Objective: Binary Cross-Entropy (BCE) with sigmoid.
- Training: AdamW, lr=1e-5, 5 epochs, default batch size 32.
- Thresholding: Fixed 0.5 per label for prediction.
- Result (dev): Accuracy ≈ 0.907, MCC ≈ 0.073 (high accuracy from predicting negatives; poor MCC due to rare classes).

Diagnosis: Labels are extremely imbalanced. BCE with a global 0.5 threshold under-predicts rare types, inflating accuracy while suppressing MCC.

#### 3) Imbalance-aware Objectives
We implemented and compared two recent multi-label long-tail techniques, adapting them to our task and pairing them with calibration and threshold tuning:

##### **3.1 Asymmetric Loss (ASL)**
Idea: Penalize false positives and false negatives asymmetrically by using separate focusing parameters (γ⁺ for positives, γ⁻ for negatives) and easy-negative clipping to reduce the loss contribution of trivially easy negatives.

Implementation details:
- We compute sigmoid probabilities p = σ(logits).
- Easy-negative clipping: clamp 1−p for negatives by a small clip (e.g., 0.1) to reduce dominance of easy negatives.
- Focusing: apply (1−p)^{γ⁺} on positive terms and p^{γ⁻} on negative terms.
- Our defaults: γ⁺=0.0, γ⁻=2.0, clip=0.1.
- Why: Encourages recall on rare positives without letting abundant negatives overwhelm gradients.

##### **3.2 Distribution-Balanced Loss (DBLoss)**
Idea: A re-balanced per-instance, per-class weight derived from inverse class frequency + Negative-Tolerant Regularization (NTR) that shifts logits by a class-dependent bias to avoid over-penalizing negatives. Optionally, a focal factor (γ) can be added.

Implementation details:
- Compute class counts on the train split only; derive inverse frequencies and a re-balanced factor r̂ per (sample, class).
- NTR: Add a learnable bias from class priors (via log-odds) scaled by κ; also scale the negative term by λ to be more tolerant.
- Our defaults that worked well: alpha=0.1, beta=10.0, mu=0.9, lam=2.0, kappa=0.05, gamma=0.0 (DB-0FL).
- Why: Explicitly counters the frequency bias and reduces negative over-confidence.

#### 4) Calibration & Decision Rules
##### **4.1 Prior-aware Logit Adjustment**
We adjust logits at inference (and optionally validation) by subtracting τ * logit(prior) where prior = P(y=1) from training data.

- Effect: Down-weights frequent classes and up-weights rare ones without touching learned weights.
- Setting: We used τ=0.5 in our best configurations.

##### **4.2 Per-class Threshold Tuning (MCC-oriented)**
Instead of using a global 0.5, we grid-search thresholds per class on the dev set to maximize per-class MCC, then apply these thresholds to test.
- Grid: t ∈ [0.02, 0.98], 49–61 steps.
- Why: Each label has a different prior and score distribution; per-class tuning aligns the decision rule with MCC.

#### 5) Sampling Strategy (Instance Level)
We complement the loss functions with a WeightedRandomSampler that upsamples instances containing rare labels:
- Compute per-instance weight as the sum of inverse class priors of the labels present in that instance.
- Use as a sampler in DataLoader (with replacement) to give batches more rare positives without distorting per-class thresholds.

#### 6) Model Architecture & Encoding
- Backbone: BartModel from facebook/bart-large (encoder states).
- Pooling: Masked mean pooling over token embeddings to form a sentence-pair representation; more stable than using the first token for BART.
- Head: Linear layer → 26 logits.
- Why this design: Keeps the backbone unchanged, light-weight head, and avoids generation; suitable for multi-label classification.

#### 7) Training Procedure
- Optimizer: AdamW (lr tuned around 1e-5 in final runs).
- Epochs: 5 (longer for the final best models after HPO).
- Loss: ASL or DBLoss, as configured.
- Evaluation each epoch: Dev accuracy & dev MCC with fixed 0.5 to track learning dynamics (not our final metric).
- Final reporting: After training, apply logit adjustment and per-class threshold tuning, report Dev (threshold-tuned) accuracy & MCC, then generate test predictions using tuned thresholds.

### Experiments
#### Exp-0 — Baseline (BCE, fixed 0.5)
Goal. Establish a reference without imbalance handling.

Setup. BART encoder → masked-mean pooling → linear head (26). Loss: BCE. AdamW lr=1e-5, 5 epochs, batch=32. Prediction threshold=0.5 per class.

Change vs. baseline. N/A (this is the baseline).

Eval. Dev metrics each epoch at fixed 0.5.

Results (dev, epoch-5). Accuracy ≈ 0.907, MCC ≈ 0.073.

Discussion. High accuracy but low MCC: the model over-predicts negatives and misses rare labels—classic long-tail failure.

#### Exp-1 — ASL (Asymmetric Loss)
Goal. Improve rare-label recall by down-weighting easy negatives and decoupling positive/negative focusing.

Setup. Same model; ASL with ypos = 0.0, yneg = 2.0, clip=0.1. AdamW lr=1e-5, 5 epochs. Optionally: WeightedRandomSampler, logit adjustment (τ=0.5), per-class threshold tuning on dev to maximize MCC.

Change vs. baseline. Replace BCE with ASL; add (optional) sampler + calibration + threshold tuning.

Eval:

- During training: dev metrics @ 0.5 (diagnostic).
- After training: Dev (threshold-tuned) using per-class thresholds; optionally with τ.

Results (representative; two runs shown to illustrate variance):

- Run A: epoch-5 MCC 0.340 (0.5); Dev (threshold-tuned) MCC 0.434.
- Run B: epoch-5 MCC 0.276 (0.5); Dev (threshold-tuned) MCC 0.382; fixed-0.5 MCC 0.118.

Discussion. ASL consistently boosts MCC vs. BCE by: (i) suppressing easy negatives (clip), (ii) focusing gradients on hard examples (γ⁻), (iii) benefiting from threshold tuning. The tuned MCC substantially exceeds the raw 0.5 MCC, confirming threshold calibration is critical in multi-label imbalance. Across runs, ASL > DBLoss on tuned MCC (in our setting), though absolute values vary with randomness/hyper-params.

#### Exp-2 DBLoss (Distribution-Balanced Loss)
Goal. Counter class-frequency bias with re-balanced per-instance weights and negative-tolerant regularization (NTR).

Setup. Same model; DBLoss with α=0.1, β=10.0, μ=0.9, λ=2.0, κ=0.05, γ=0.0 (DB-0FL). AdamW lr=1e-5, 5 epochs. Optionally: WeightedRandomSampler, logit adjustment (τ=0.5), per-class threshold tuning.

Change vs. baseline. Replace BCE with DBLoss; derive class counts on train split; add calibration + threshold tuning.

Eval. Same protocol as ASL.

Results:

- epoch-5 MCC 0.204 (0.5); Dev (threshold-tuned) MCC 0.323; fixed-0.5 MCC 0.157.

Discussion. DBLoss improves over BCE, mainly after applying logit adjustment and per-class thresholds. It trails ASL in our experiments, likely because ASL’s easy-negative clipping + γ⁻ more aggressively combats the sea of negatives. DBLoss remains attractive when we want a principled, prior-aware objective with NTR.

#### Exp-3 — Calibration ablation (thresholds + logit adjustment)
Goal. Quantify how much MCC comes from decision calibration vs. loss alone.

Setup. Take trained ASL/DBLoss checkpoints; evaluate:

- fixed 0.5 thresholds;
- per-class tuned thresholds;
- tuned thresholds + logit adjustment (τ=0.5).

Change vs. baseline. Post-hoc calibration only.

Results (qualitative trend from your logs).

- For both losses, tuned thresholds increase MCC by a large margin vs. 0.5.
- Logit adjustment (τ=0.5) further improves tuned MCC (e.g., DBLoss: 0.296 → 0.323; ASL: 0.382–0.434 in earlier run).

Discussion. In long-tail multi-label settings, decision rules matter almost as much as the loss. Per-class thresholds align precision/recall trade-offs per label; τ corrects head/tail priors at inference.

#### Exp-4 — WeightedRandomSampler ablation
Goal. Test if up-sampling rare-label instances helps MCC.

Setup. Enable per-instance weights = sum of inverse class priors of labels present; use WeightedRandomSampler with replacement during training, keeping loss and other settings fixed.

Change vs. previous. Sampling only.

Results. Small but consistent MCC gains when combined with ASL/DBLoss, especially when positive labels per batch are scarce.

Discussion. The sampler increases exposure to rare positives without fabricating data; best used alongside ASL/DBLoss + calibration.

#### Key findings & trends
- Loss matters, but thresholds matter too. ASL/DBLoss both beat BCE; per-class threshold tuning and τ-adjustment deliver large additional MCC gains.
- ASL > DBLoss (here). With γ⁻ and clipping, ASL achieved the highest tuned MCC in our experiments (e.g., 0.382–0.532 vs. DBLoss 0.296–0.323), though both are strong vs. baseline.
- Samplers help when positives per batch are scarce; effect is additive with loss and calibration.
- Accuracy is misleading. Fixed-0.5 accuracy stays high while MCC changes significantly—reinforcing MCC as the right objective in long-tail multi-label detection.

What didn’t help (or helped less)
- Training longer without calibration raised accuracy but did not reliably improve MCC.
- Using a global threshold (0.5) after fancy losses still under-performed per-class tuned thresholds.
- Naive data augmentation was not attempted due to label-drift risk for fine-grained types; future work could explore type-preserving augmentation with verification.

### **Results**

All runs are 5 epochs on the same train/dev split.

Summary table:

| Method            | Loss Params                                                                 | Extras                                        | Dev Acc (0.5) | Dev MCC (0.5) | Dev Acc (tuned) | Dev MCC (tuned) |
|-------------------|------------------------------------------------------------------------------|-----------------------------------------------|:-------------:|:-------------:|:---------------:|:---------------:|
| **Baseline (BCE)**| —                                                                            | token-0 pooling, fixed 0.5 thresholds         |    0.907      |    0.073      |       —         |       —         |
| **DB Loss**       | α=0.1, β=10, **μ=0.9**, λ=2.0, κ=0.05, γ=0.0 (DB-0FL)                        | **WeightedRandomSampler**, logit-adj **τ=0.5**|    0.770      |    0.157      |     **0.780**   |     **0.323**   |
| **ASL**           | γ_pos=0.0, **γ_neg=2.0**, **clip=0.1**                                       | **WeightedRandomSampler**, logit-adj **τ=0.5**|    0.346      |    0.107      |     **0.818**   |     **0.371**   |
| **ASL (HPO best)**| γ_pos=0.0, γ_neg=1.0, clip=0.1633; lr=7.0e-6; wd=5.94e-4; dropout=0.161; batch=16 | WeightedRandomSampler, per-class thresholds (61 steps), logit-adj τ=0.2773|    -     |    -      |     **0.873**   |     **0.532**   |


**Notes**:
- “Fixed 0.5” uses a single threshold for every label. “Threshold-tuned” picks a threshold per label to maximize dev MCC (after optional logit adjustment).
- DB Loss: raw epoch-5 MCC (fixed 0.5) rose from ~0.001 → 0.204 during training; with calibration (logit adjust τ=0.5 + per-class thresholds) it reached 0.323.
- ASL: raw epoch-5 MCC (fixed 0.5) reached 0.276; with calibration (τ=0.5 + per-class thresholds) it reached 0.382.
- Also it can be noted that there is a comprimise between the accuracy and the MCC, as the model that predicts the same type predicts high accuracy (most labels are negatives) but near-zero MCC—i.e., it completely fails rare types. So lower accuracy with higher MCC usually means you’re finally catching rare positives instead of guessing “negative” everywhere.

#### Takeaways
- Loss matters: Replacing BCE with ASL or DB increases tail recall and MCC markedly.
- Calibration matters even more: Per-class thresholds + τ-adjustment roughly doubled MCC vs fixed 0.5.
- ASL > DB on our data: ASL delivered the best tuned MCC (0.382). This aligns with reports that ASL excels when negative dominance is the main bottleneck. 
- DB is competitive and theoretically appealing when label co-occurrence is severe; EMNLP-21 shows DB’s advantages for multi-label text, which we partially observe (MCC 0.323 vs baseline 0.073).

### Hyperparameter Optimization 
Best params found (ASL, from Optuna)
- lr=7.04e-06, weight_decay=5.94e-04, dropout=0.161
- gamma_pos=0.0 (implicit), gamma_neg=1.0
- clip=0.1633
- tau=0.2773
- batch_size=16, th_grid_steps=61
- use_weighted_sampler=True

Why these may have worked:
- Moderate τ (0.28): just enough prior correction without over-flattening head classes.
- High clip (~0.16): aggressively downweights easy negatives, which can boost recall on rare labels.
- Lower γ⁻ (=1) with a higher clip is a reasonable trade-off: clipping is already suppressing the sea of easy negatives; you need less focal pressure.
- Higher weight decay + dropout 0.16 likely improved generalization for tail labels.

After hyper-paramter optimization achieved a tuned Dev accuracy of `0.873` and tuned MCC of `0.532`

### Visualization
**ASL**:

Dev accuracy:

<img width="640" height="480" alt="dev_acc_0p5" src="https://github.com/user-attachments/assets/cdbb535f-4b3c-4faa-bdb5-356f0a429478" />


Train loss:

<img width="640" height="480" alt="train_loss" src="https://github.com/user-attachments/assets/9ea7baa1-336d-4e3e-b671-ea5009170247" />


Dev MCC:

<img width="640" height="480" alt="dev_mcc_0p5" src="https://github.com/user-attachments/assets/dca275f3-6d0f-4a51-bc47-d5761b181858" />

**DB loss**:

Dev accuracy:

<img width="640" height="480" alt="dev_acc_0p5" src="https://github.com/user-attachments/assets/8db26460-7b47-4c37-aeb5-c52b27ff5b8c" />


Train loss:

<img width="640" height="480" alt="train_loss" src="https://github.com/user-attachments/assets/bc6321f4-1a51-4d98-bb07-9dbcbf1bb7e0" />


Dev MCC:

<img width="640" height="480" alt="dev_mcc_0p5" src="https://github.com/user-attachments/assets/83c76ae7-1fd5-4d78-8541-78645d096dec" />



#### What the curves show
##### Train loss (both ASL & DBLoss)
- Monotonic decrease across epochs → the models are optimizing stably.
- DBLoss train loss starts smaller (because of re-balancing and NTR) and drops gently; ASL’s absolute scale is higher but steadily declines as well.

##### Dev accuracy @ 0.5 (ASL & DBLoss)
- ASL: slight U-shape (↓ until epoch 3, then ↑); ends lower than it started.
- DBLoss: mostly flat high (~0.895) with a dip near epoch 4.
- Interpretation: as models become less conservative and predict more positives, accuracy can fall even while MCC improves (more rare positives flagged → more false positives under a single global threshold).

##### Dev MCC @ 0.5 (ASL & DBLoss)
- ASL: clear, steady increase each epoch (≈0.02 → ≈0.28 by epoch 5).
- DBLoss: slower start, then sharper gains late (≈0.00 → ≈0.23–0.24 by epoch 5).
- Interpretation: both objectives improve the balance of precision/recall across classes; ASL converges earlier and ends slightly higher in this run.

Side-by-side insights:

**Convergence speed:**
- ASL improves MCC earlier (by epoch 3) and more smoothly.
- DBLoss catches up later; its MCC curve is more convex (bigger late-epoch gains).

**Final performance (raw 0.5):**
- ASL > DBLoss on dev MCC in these plots.
- Dev accuracy differences are small and not the deciding factor.

**Why accuracy & MCC disagree:**
- With many negatives, accuracy rewards conservative predictions; MCC rewards correctly identifying rare positives as well. As training progresses, models trade a little accuracy for much better MCC—exactly what we want.


#### References
- Asymmetric Loss (ASL): Ridnik et al., Asymmetric Loss for Multi-Label Classification, ICCV 2021. [CVF open access](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf)
- Distribution-Balanced Loss (DB): Wu et al., Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets, ECCV 2020 (and arXiv). [ECVA](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490154.pdf), [arXiv](https://arxiv.org/abs/2007.09654)
- DB for multi-label text: Huang et al., Balancing Methods for Multi-Label Text Classification with Long-Tailed Class Distribution, EMNLP 2021. [ACL Anthology](https://aclanthology.org/2021.emnlp-main.643/)
- Class-Balanced Loss (background): Cui et al., CVPR 2019. [CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)

## 5) Bart paraphrase type generation:
### 1) Install extra packages
```bash
pip install optuna
```
### 2) Project structure & data
Your repo should look like this:
```bash
pip install optuna
```project_root/
├─ bart_generation.py                  
├─ data/
│  ├─ etpc-paraphrase-train.csv
│  └─ etpc-paraphrase-generation-test-student.csv
├─ predictions/
│  └─ bart/                            # will be created if you generate test outputs
└─ ...
```
### 3) Execution modes
#### A) Supervised-only (MLE)
Trains BART with padding-masked cross-entropy and saves the best dev checkpoint to best_supervised_bart_model.pt.
```bash
python bart_generation.py \
  --do_supervised \
  --supervised_epochs 8 \
  --seed 11711
```
After this, later runs will automatically start from `best_supervised_bart_model.pt` if present.

#### B) RL (SCST) on penalized BLEU (starting from best supervised if available)
Fine-tunes with self-critical sequence training using the penalized BLEU reward.
```bash
python bart_generation.py \
  --use_gpu \
  --use_rl --rl_epochs 5 \
  --rl_lr 2e-6 --rl_lambda 0.3 --rl_batch_size 8 --rl_max_len 50 \
  --seed 11711
```
#### C) Quality-guided RL (optional)
Adds a controllable reward blend (semantic, syntactic, lexical). Example weights:
```bash
python bart_generation.py \
  --use_gpu \
  --use_rl --rl_epochs 6 \
  --quality_weights "0.7 0.2 0.1" \
  --rl_lr 2e-6 --rl_lambda 0.3 --rl_batch_size 8 --rl_max_len 50 \
  --seed 11711
```
#### D) Hyperparameter Optimization (HPO) + RL
Runs Optuna to tune RL hyperparameters on the dev penalized BLEU, then applies the best params for a full RL run.
```bash
python bart_generation.py \
  --use_gpu \
  --do_supervised --supervised_epochs 8 \
  --use_rl --rl_epochs 5 \
  --hpo --hpo_trials 20 --hpo_rl_epochs 2 \
  --seed 11711
```
### 4) CLI quick reference
```bash
--do_supervised            Run supervised MLE training before RL
--supervised_epochs        Epochs for supervised phase (default: 10)

--use_rl                   Enable RL (SCST) fine-tuning
--rl_epochs                RL epochs (default: 2)
--rl_lr                    RL AdamW learning rate (default: 2e-6)
--rl_lambda                Weight for RL vs MLE blend (default: 0.3)
--rl_max_len               Max generation length during RL (default: 50)
--rl_batch_size            Batch size for RL (default: 8)

--quality_weights          e.g., "0.7 0.2 0.1" (semantic, syntactic, lexical)

--hpo                      Run Optuna HPO before final RL
--hpo_trials               Number of trials (default: 20)
--hpo_rl_epochs            RL epochs per trial (short budget; default: 1)
--hpo_timeout              Optional global timeout for HPO (seconds)

--use_gpu                  Use CUDA if available
--seed                     Global random seed (default: 11711)
````

# Methodology

This task focuses on improving paraphrase generation performance using a customized BART architecture. Our primary goal is to explore, implement, and evaluate various training strategies and reward functions within a unified and extensible pipeline that supports both baseline and advanced methods via configurable command-line arguments. The overarching methodology is to systematically extend the standard BART model into a flexible research platform for paraphrase generation tasks, incorporating reinforcement learning-based fine-tuning, quality-guided rewards, and hyperparameter optimization. Through this approach, we aim to enhance the diversity and semantic fidelity of generated paraphrases while maintaining alignment with evaluation metrics such as penalized BLEU.
## Problem & Objective

We tackle **paraphrase generation** with the goal of producing fluent outputs that **preserve meaning** while **discouraging copying** from the source. Training is aligned with evaluation using a **penalized BLEU** objective, used both as the RL reward and for dev model selection:

$$
R(\text{ref}, \text{inp}, \text{hyp}) = \text{BLEU}(\text{ref}, \text{hyp}) \times \frac{100 - \text{BLEU}(\text{inp}, \text{hyp})}{52}
$$

We compute BLEU with **SacreBLEU** (`effective_order=True`) to mitigate length and low-count artifacts.


## Dataset
- We were provided with etpc-paraphrase-train.csv for training data and etpc-paraphrase-generation-test-student.csv as test data.
- We split the training data into 80/20 to obtain the dev data.


## Model & Data Encoding

**Backbone:** `facebook/bart-large` (seq2seq)
**Inputs:** Each example is serialized as
```ini
encoder_input = sentence1  <SEP>  sentence1_segment_location  <SEP>  paraphrase_type_ids
```
where `<SEP>` is the tokenizer’s sep_token (or </s> fallback).

- **Targets:** `sentence2` (gold paraphrase).

- **Padding masking:** target pads are ignored via labels `[labels == pad_id] = -100`, ensuring <pad> tokens do not contribute to the cross-entropy, otherwise there will be noise in the loss.

## Baseline: Supervised Fine-Tuning (MLE)
We first fine-tune BART with standard cross-entropy on (input → target) pairs. This yields a fluent, stable initialization:
- Optimizer: **AdamW**
- Checkpointing: save `best_supervised_bart_model.pt` whenever dev penalized BLEU improves.
- Rationale: MLE provides strong fluency and stabilizes the subsequent RL phase.

## Improvement: Self-Critical Sequence Training (SCST) on Penalized BLEU
We adopt **SCST** to directly optimize our task reward. For each batch:

- **Greedy baseline (`greedy_decode`)**: decode deterministically (no sampling).  
- **Sampled rollout (`sample_decode`)**: decode with exploration  
  `do_sample=True, top_p=0.9, top_k=50, temperature=1.5, min_length=8, repetition_penalty=1.1`.  
- **Per-sample reward**: sentence-level SacreBLEU with smoothing composed into penalized BLEU as above.  
- **Advantage**:
   
  **_A = R_sample - R_greedy_**

- **Policy gradient**: compute $\log p_\theta(y_{\text{sample}} \mid x)$ via teacher forcing (`sequence_logprob`) and minimize **_-A.logp_**.     

#### Stability controls
- **Loss mixing (MLE stabilizer):**  
  _L = λ_rl · L_PG + (1 − λ_rl) · L_MLE,**λ_rl = 0.3**_
- **Advantage normalization** with variance floor (`max(std, 0.1)`); skip PG if `std < 1e-6`.  
- **Gradient clipping:** `clip_grad_norm_(…, 1.0)`.  
- **Optimizer state persistence:** a single AdamW optimizer is created once in the driver and passed into each RL epoch so momentum/second-moment statistics carry across epochs.  

#### Why it helps
- The reward aligns directly with how we evaluate (faithfulness × novelty), explicitly penalizing copies.  
- SCST’s **self-baseline** reduces variance by subtracting the greedy trajectory’s reward.  

---

### Optional: Quality-Guided Reward
Inspired by [Bandel et al., 2022], we add an **optional quality-guided reward**:  

- **Semantic similarity** (SacreBLEU vs. reference, proxy).  
- **Syntactic variation** (length-difference proxy).  
- **Lexical variation** (overlap penalty proxy).  

Users set weights via `--quality_weights "sem syn lex"` or a JSON config.  
This keeps training lightweight (no extra models) while enabling interpretable trade-offs between adequacy and diversity.  

---

### Hyperparameter Optimization (HPO) for RL (Optional)
We include an Optuna (TPE) search that maximizes dev penalized BLEU over RL-critical knobs:

- **Search space**  
  - `rl_lr` ∈ [1e-7, 5e-5] (log)  
  - `lambda_rl` ∈ [0.1, 0.7]  
  - `rl_max_len` ∈ {40, 50, 60, 70, 80}  
  - `rl_batch_size` ∈ {4, 8, 16}  

- **Protocol**  
  - Each trial runs short RL (`--hpo_rl_epochs`, default 1) on a deterministic subset of train for speed.  
  - Model is reset to the pre-HPO snapshot before each trial.  
  - We evaluate on the same dev split and let Optuna maximize the score.  
  - After HPO, we reset to the supervised snapshot, apply best params, and run full RL for `--rl_epochs` on the full train set.  

---

### Evaluation & Model Selection
- **Metric**: dev penalized BLEU (SacreBLEU, `effective_order=True`).  
- **Consistency**: same BLEU flavor for reward (sentence-level) and reporting (corpus-level) to reduce metric drift.  
- **Checkpointing**: save `best_bart_rl_model.pt` whenever dev penalized BLEU improves during RL; reload the best at the end.  

---

### Relation to Prior Work & Our Contribution
- **Li et al., 2018 — Paraphrase Generation with Deep RL.**  
  They train a generator plus a learned evaluator and fine-tune with the evaluator’s signal.  
  We simplify this by using SCST with a penalized BLEU reward—no extra evaluator to train—yielding a lean, reproducible pipeline that still optimizes a task-aligned objective.  

- **Bandel et al., 2022 — Quality-Guided Paraphrase Generation.**  
  They introduce explicit control along semantic/syntactic/lexical axes.  
  We adopt this controllability idea with a lightweight reward blend (BLEU proxy, length-variation proxy, lexical-overlap proxy) selectable via `--quality_weights`, integrating seamlessly into SCST without additional models.  

**Net contribution:**  
A practical, stable paraphrase system that:  
1. Aligns training with evaluation via penalized BLEU.  
2. Stabilizes RL with MLE mixing and variance guards.  
3. Offers optional controllability with a simple, interpretable reward blend.  
4. Includes reproducible HPO for RL-sensitive knobs.  

---

### Limitations & Design Trade-offs
- Sentence-level BLEU is a proxy for semantics; excessive exploration can still drift meaning.  
- Penalized BLEU can oscillate across RL epochs; early stopping and best-checkpoint selection mitigate this.  
- The quality-guided proxies are lightweight (not learned quality models); they trade precision for simplicity and speed.  

---

### Where to Find Things in Code
- **Data & collation**: `transform_data`, `ParaphraseRLDataset`, `rl_transform_data`  
- **Decoding**: `greedy_decode`, `sample_decode`  
- **Rewards**: `penalized_bleu_per_sample`, `quality_guided_reward`  
- **PG log-probs**: `sequence_logprob`  
- **Training loops**: `train_model` (MLE), `rl_finetune_epoch` (SCST)  
- **Orchestration & HPO**: `finetune_paraphrase_generation` (flags: `--use_rl`, `--quality_weights`, `--hpo`, `--hpo_*`, `--rl_*`)

# Experiments

In this section, we describe the key experiments executed during the development of our paraphrase generation model using reinforcement learning (RL) and Self-Critical Sequence Training (SCST). The experiments aim to evaluate the performance of different models and hyperparameter configurations.

#### 1. **Baseline Model: BART with Supervised Learning (MLE)**

**Task**: Paraphrase Generation  
**Model**: BART (Pre-trained)  
**Method**: Supervised training (Maximum Likelihood Estimation - MLE)  
**Hyperparameters**:  
- Epochs: 7  
- Batch Size: 8  
- Learning Rate: 1e-5  

**Expectations**:  
The baseline model is expected to generate paraphrases but will be limited by the standard MLE training, hence it is expected to copy the output from the input sentence, resulting in a low penalized BLEU score.

**Initial Improvement**:  
Initially, we performed some tweaking to the `transform` function for the baseline model. Specifically, we implemented **padding masking** where target pads are ignored during the loss computation by setting `labels[labels == pad_id] = -100`. This ensures that tokens that are padding do not contribute to the cross-entropy loss, preventing noise in the training process. 

**Changes Compared to Previous Model**:  
- Introduced padding masking to ensure that padded tokens are excluded from loss calculations, thus improving training efficiency and reducing noise.

**Results**:  
After this improvement in the baseline model, we observed a significant performance increase, achieving a **26.21 penalized BLEU score** on the development dataset, which reflects a reasonable starting point for paraphrase generation.

| Epoch | Loss     | Average Loss | BLEU Score | Negative BLEU Score | Penalized BLEU | Dev BLEU Score |
|-------|----------|--------------|------------|---------------------|----------------|----------------|
| 1     | 1.4742   | 1.5799       | 48.06      | 2.71                | 2.51           | 2.51           |
| 2     | 1.0772   | 1.1527       | 46.93      | 8.49                | 7.66           | 7.66           |
| 3     | 1.1725   | 0.9554       | 46.98      | 10.80               | 9.76           | 9.76           |
| 4     | 0.8185   | 0.7456       | 45.95      | 19.08               | 16.85          | 16.85          |
| 5     | 0.4700   | 0.5236       | 42.95      | 27.81               | 22.97          | 22.97          |
| 6     | 0.3281   | 0.3323       | 43.01      | 29.48               | 24.38          | 24.38          |
| 7     | 0.2525   | 0.2159       | 42.29      | 32.23               | 26.21          | 26.21          |


#### 2. **SCST with Penalized BLEU Reward**

**Task**: Paraphrase Generation  
**Model**: BART + SCST (Self-Critical Sequence Training)  
**Method**: Reinforcement Learning with Penalized BLEU Reward  
**Hyperparameters**:  
- RL Epochs: 12  
- Learning Rate: 2e-6  
- Lambda RL: 0.3 (mixing with MLE)  
- Batch Size: 8  

**Expectations**:  
We expect SCST to improve the model's ability to generate paraphrases by aligning training directly with the evaluation metric (penalized BLEU). SCST should help the model explore a wider range of paraphrases and discourage overfitting to the training data. By directly optimizing penalized BLEU, the model can focus on generating more diverse and faithful paraphrases.

**Changes Compared to Previous Model**:  
- Introduction of reinforcement learning for fine-tuning after initial MLE training.
- Reward function: Penalized BLEU score with direct feedback from a comparison between greedy decoding and sampled outputs. This incentivizes better exploration of paraphrasing possibilities.
- Introduced **smoothing by BLEU** (`effective_order=True`) to reduce length and low-count artifacts in BLEU score computation, improving the robustness of evaluation.

**Evaluation Metrics**:  
- **Dev Penalized BLEU**  
- **Sampled vs. Greedy Rewards** (Evaluation of exploration vs. exploitation)

#### STSC-Focused RL Training (12 Epochs)

| RL Epoch | BLEU (ref→hyp) | 100 − BLEU (input→hyp) | Penalized BLEU |
|---|---:|---:|---:|
| 1 | 39.04 | 38.15 | 28.65 |
| 2 | 42.21 | 30.65 | 24.88 |
| 3 | 38.13 | 40.46 | 29.67 |
| 4 | 38.08 | 40.94 | 29.98 |
| 5 | 31.37 | 50.51 | 30.47 |
| 6 | 34.57 | 44.89 | 29.85 |
| 7 | 42.45 | 29.67 | 24.22 |
| 8 | 38.42 | 39.52 | 29.20 |
| 9 | 39.36 | 38.25 | 28.95 |
| 10 | 35.02 | 47.18 | 31.77 |
| 11 | 37.99 | 39.32 | 28.72 |
| 12 | 32.84 | 50.13 | 31.66 |

- **Best Penalized BLEU:** 31.77 (Epoch 10)  
- **Improvement:** 26.21 → 31.77 from start of RL phase  
- **Observation:** Penalized BLEU fluctuates due to trade-off between fidelity to input and semantic divergence.


#### 3. **Quality-Guided Reward with Optional Weights**

**Task**: Paraphrase Generation  
**Model**: BART + SCST with Quality-Guided Reward  
**Method**: Reinforcement Learning with Quality-Guided Reward (Semantic, Syntactic, Lexical)  
**Hyperparameters**:  
- **Quality Weights**: "sem syn lex" (set via command-line arguments)  
- **RL Epochs**: 12  
- **Learning Rate**: 2e-6  
- **Lambda RL**: 0.3  
- **Batch Size**: 8  

**Expectations**:  
By introducing explicit control over semantic, syntactic, and lexical variations in the paraphrase generation, we anticipate the model will produce more diverse and semantically faithful paraphrases. This improvement is expected to address areas where the standard SCST-based model might struggle, particularly in maintaining a balance between adequacy and diversity.

**Changes Compared to Previous Model**:  
- **Additional control** for generation quality via user-defined weights on semantic, syntactic, and lexical factors.
- This allows the user to fine-tune the importance of different aspects of paraphrase generation, providing a more flexible approach to handling quality.

**Evaluation Metrics**:  
- **Dev Penalized BLEU**  
- **Quality Evaluation** (Semantic, Syntactic, Lexical)

| **Phase/Epoch** | **BLEU (ref→hyp)** | **100 − BLEU (input→hyp)** | **Penalized BLEU** |
|-----------------|--------------------|----------------------------|--------------------|
| RL Epoch 1      | 42.36              | 30.73                      | 25.04              |
| RL Epoch 2      | 25.59              | 58.04                      | 28.56              |
| RL Epoch 3      | 39.67              | 36.20                      | 27.61              |
| RL Epoch 4      | 42.18              | 30.55                      | 24.78              |
| RL Epoch 5      | 35.69              | 39.17                      | 26.88              |
| RL Epoch 6      | 35.31              | 41.57                      | 28.23              |
| RL Epoch 7      | 41.47              | 32.24                      | 25.71              |
| RL Epoch 8      | 36.71              | 42.05                      | 29.69              |
| RL Epoch 9      | 38.39              | 39.22                      | 28.96              |
| RL Epoch 10     | 41.38              | 31.11                      | 24.75              |
| RL Epoch 11     | 34.07              | 43.58                      | 28.56              |
| RL Epoch 12     | 43.67              | 24.97                      | 20.97              |

**RL Improved BLEU**: 26.21 → 29.69


#### 4. **Hyperparameter Optimization (HPO) with Optuna**

**Task**: Hyperparameter Tuning for SCST  
**Method**: Hyperparameter Optimization using Optuna with TPE (Tree-structured Parzen Estimator)  
**Hyperparameters to Optimize**:  
- **rl_lr** (learning rate for RL)  
- **lambda_rl** (weight between RL and MLE)  
- **rl_max_len** (maximum sequence length)  
- **rl_batch_size** (batch size)  

**Expectations**:  
The objective of this experiment is to find the optimal combination of hyperparameters that maximizes the penalized BLEU score on the development set. By doing so, we aim to improve model performance and training efficiency by identifying the best settings for reinforcement learning-related hyperparameters.

**Changes Compared to Previous Model**:  
- Introduction of an **automatic hyperparameter search** with Optuna, enabling the identification of the best RL-related hyperparameters without requiring manual tuning.
- This process aims to streamline the search for the most effective settings for RL fine-tuning.

**Evaluation Metrics**:  
- **Dev Penalized BLEU**  
- **Model performance** based on the best hyperparameters identified through the HPO process.

  **Results**:
- For each trial, we run short RL (`--hpo_rl_epochs`, 2)
  ### Hyperparameter Optimization Results

### Hyperparameter Optimization Results (Top 5 Trials Based on Highest Penalized BLEU)

| Trial | RL Learning Rate (rl_lr) | Lambda RL (lambda_rl) | Max Length (rl_max_len) | Batch Size (rl_batch_size) | Penalized BLEU Score |
|-------|--------------------------|------------------------|--------------------------|---------------------------|----------------------|
| 0     | 2.88e-06                 | 0.3587                 | 80                       | 4                         | 30.95               |
| 10    | 6.67e-06                 | 0.2913                 | 40                       | 4                         | 29.03               |
| 16    | 1.27e-06                 | 0.3211                 | 80                       | 16                        | 28.55               |
| 24    | 4.11e-06                 | 0.3640                 | 40                       | 4                         | 28.42               |
| 20    | 1.19e-06                 | 0.4693                 | 60                       | 4                         | 28.62               |

### Best Hyperparameters:
- **Best Trial**: Trial 0
- **Parameters**: 
  - **rl_lr**: 2.88e-06
  - **lambda_rl**: 0.3587
  - **rl_max_len**: 80
  - **rl_batch_size**: 4
  - **Penalized BLEU Score**: 30.95
    
- It can be interpreted from that table that batch size = 4 and learning rate in e-06 gives better results.



### Best Hyperparameters:
- **Best Trial**: Trial 0
- **Parameters**: 
  - **rl_lr**: 2.88e-06
  - **lambda_rl**: 0.3587
  - **rl_max_len**: 80
  - **rl_batch_size**: 4
- **Best Penalized BLEU Score**: 30.95

<img width="1234" height="608" alt="image" src="https://github.com/user-attachments/assets/0979a4a2-4756-4cdd-a37c-80323bcee2d5" />

#### Two-Phase RL Training with Weighted Quality and STSC

We implemented a two-phase reinforcement learning (RL) training strategy to further optimize paraphrase generation:

1. **Phase 1 – Quality-Focused RL:**  
   - Trained for **10 epochs** using weighted rewards `[0.7, 0.2, 0.1]` corresponding to **semantic similarity, syntactic variation, and lexical variation**.  
   - This phase prioritized high-quality paraphrase generation while maintaining syntactic and lexical diversity.

2. **Phase 2 – STSC-Focused RL:**  
   - Loaded the resulting weights from Phase 1 into the model and performed **10 more RL epochs** targeting **STSC (sentence-to-sentence consistency)**.  
   - All epochs in both phases used **optimized hyperparameters** identified through prior HPO experiments.

**Results – Penalized BLEU:**

| Phase/Epoch | BLEU (ref→hyp) | 100 − BLEU (input→hyp) | Penalized BLEU |
|---|---:|---:|---:|
| RL Epoch 1 | 43.756 | 29.713 | 25.002 |
| RL Epoch 2 | 40.295 | 39.688 | 30.754 |
| RL Epoch 3 | 26.519 | 59.586 | 30.388 |
| *RL Epoch 4 (best)* | *35.857* | *48.581* | **33.500** |
| RL Epoch 5 | 35.669 | 48.321 | 33.146 |
| RL Epoch 6 | 35.994 | 47.646 | 32.980 |

**Observations and Possible Reasons for Improvement:**  
- **Better penalized BLEU:** The two-phase approach led to a **peak penalized BLEU of 33.5**, higher than any previous single-phase RL run.  
- **Quality weighting in Phase 1** allowed the model to internalize semantic fidelity while maintaining syntactic and lexical variety.  
- **Phase 2 STSC fine-tuning** ensured stronger sentence-to-sentence consistency, improving the penalization term and stabilizing BLEU scores.  
- **Cumulative effect:** Starting STSC training from weights already optimized for quality helped the model converge faster and more effectively than training from scratch.  
- Overall, **progressive optimization with phased objectives and tuned hyperparameters** produced a model that balanced creativity and accuracy, yielding higher penalized BLEU.


### Quality-Guided Reward with Optional Weights vs SCST with Penalized BLEU Reward

As expected, **SCST with Penalized BLEU Reward** performs better in terms of **Penalized BLEU** because it is specifically optimized for this metric. The SCST method directly maximizes **Penalized BLEU**, which leads to higher diversity in the paraphrases, as evidenced by the improvement from **26.21** (baseline) to **31.77**. This method focuses on reducing input similarity while ensuring that the generated output closely matches the reference.

On the other hand, **Quality-Guided Reward with Optional Weights** focuses more on enhancing **semantic similarity** by balancing various factors like **semantic**, **syntactic**, and **lexical** variations. While it achieved a **Penalized BLEU** score of **29.69**, which is lower than SCST's **31.77**, it improved the **BLEU (ref→hyp)** score to **36.71**. However, this focus on quality and balance led to higher **100 − BLEU (input→hyp)** (42.05), suggesting that the model prioritized improving reference similarity at the expense of input diversity.

In summary, **SCST** is tailored to improve **Penalized BLEU**, making it more effective for tasks requiring diversity and reduced input similarity. **Quality-Guided Reward**, on the other hand, emphasizes semantic quality, making it better suited for tasks where reference alignment is critical, albeit with a trade-off in diversity.
  

  ## Key Takeaways

### 1. **Change in Numbers During Generation**  
One notable observation during the evaluation of the paraphrase generation was the change in numbers within the generated sentences. Specifically, the numerical values in the original input sentence were altered in the paraphrases, even when they were expected to remain unchanged. This issue likely stemmed from the lack of high-quality paraphrasing in the model output. We hypothesize that the model’s inability to correctly retain key numerical information and other sensitive tokens might have been caused by suboptimal quality in the generated paraphrases, which are unable to preserve such specific details.

### 2. **Trade-off Between Penalized BLEU and BLEU Score**  
We noticed a consistent trend where an increase in the **Penalized BLEU score** was accompanied by a decrease in the **BLEU score**. This suggests that there exists a trade-off between these two metrics. The **BLEU score** is a measure of similarity to the reference, but penalized BLEU tries to mitigate issues where the generated text overly matches the input (i.e., it discourages repetition of the input text). Thus, models that generate highly similar paraphrases to the reference might achieve high BLEU scores but lower penalized BLEU scores, and vice versa.

### 3. **Trade-off Between Quality-Generated and Penalized BLEU Score**  
A similar trade-off was observed between the **Quality-Generated Paraphrases** and the **Penalized BLEU Score**. In our experiments, as the **Quality-Generated** metric improved (using weighted rewards for semantic, syntactic, and lexical variations), the **Penalized BLEU** score also showed improvement, but the trade-off with BLEU was still evident. A model focusing heavily on quality might not always optimize for **BLEU**, as it sacrifices some level of similarity with the reference sentence in favor of generating more semantically diverse and syntactically varied output.

### 4. **Optimal Balance Between BLEU and Penalized BLEU**  
The experiments indicate that the best-performing models, in terms of **Penalized BLEU**, were able to strike a balance between maintaining a high similarity to the reference sentence while introducing enough variation to avoid the pitfalls of overly repetitive text generation. This balance is crucial for improving both the **BLEU** and **Penalized BLEU** scores simultaneously, making sure that paraphrases are both semantically correct and sufficiently distinct from the input text.


# Results 

| **Paraphrase Type Generation (PTG)**                        | **Penalized BLEU Score** | **BLEU (ref→hyp)** | **100 − BLEU (input→hyp)** |
|-------------------------------------------------------------|--------------------------|--------------------|----------------------------|
| Baseline with padding masking                               | 26.21                    | 42.29              | 32.23                      |
| SCST with Penalized BLEU Reward                              | 31.77                    | 35.02              | 47.18                      |
| Quality-Guided Reward with Optional Weights                  | 29.69                   |  36.71             |  42.05                      |
| Two-Phase RL Training with Weighted Quality and STSC        | 33.50                    | 35.85              | 48.58                      |

### Results Discussion

The **Paraphrase Type Generation (PTG)** experiments demonstrate how different training methods affect the **Penalized BLEU Score**, **BLEU (ref→hyp)**, and **100 − BLEU (input→hyp)**:

1. **Baseline with Padding Masking (26.21 Penalized BLEU)**:
   - The baseline achieved a **Penalized BLEU** score of **26.21**. The **BLEU (ref→hyp)** of **42.29** and **100 − BLEU (input→hyp)** of **32.23** reflect a moderate balance between reference similarity and input diversity.

2. **SCST with Penalized BLEU Reward (31.77 Penalized BLEU)**:
   - **SCST** improved the **Penalized BLEU** score to **31.77** but resulted in a lower **BLEU (ref→hyp)** of **35.02** and higher **100 − BLEU (input→hyp)** at **47.18**, showing a trade-off between diversity and semantic accuracy.

3. **Quality-Guided Reward with Optional Weights (29.69 Penalized BLEU)**:
   - This method yielded a **Penalized BLEU** score of **29.69**, slightly lower than **SCST**, but with a better **BLEU (ref→hyp)** of **36.71** and a **100 − BLEU (input→hyp)** of **42.05**. It demonstrates that weighted quality aspects can improve the model's semantic alignment but at the cost of diversity.

4. **Two-Phase RL Training with Weighted Quality and STSC (33.50 Penalized BLEU)**:
   - The **Two-Phase RL Training** method produced the highest **Penalized BLEU** score of **33.50**, with a **BLEU (ref→hyp)** of **35.85** and **100 − BLEU (input→hyp)** of **48.58**, striking a good balance between diversity and semantic quality.

### Observations:
**As already discussed in Experiments takeaways**
- **Trade-off Between BLEU and Penalized BLEU**: As **Penalized BLEU** increases, **BLEU (ref→hyp)** tends to decrease, indicating a trade-off between semantic accuracy and diversity.
- **Impact of Rewarding Quality Aspects**: **Quality-Guided Reward** improved reference similarity but led to a slightly lower **Penalized BLEU** than **SCST**.
- **Two-Phase RL Training**: This method, combining broader training with **STSC** fine-tuning, achieved the best balance of diversity and semantic alignment.

### Conclusion:

**Two-Phase RL Training with Weighted Quality and STSC** provides the best overall results, with the highest **Penalized BLEU** score, offering an optimal balance between semantic accuracy and paraphrase diversity.  The key takeaway from this analysis is that paraphrase generation is a complex problem that involves balancing between **retaining important details**, **avoiding repetitive outputs**, and **introducing enough variety** to produce fluent and accurate paraphrases. Achieving the best results requires careful tuning of reward functions, attention to the trade-offs between **BLEU** and **Penalized BLEU** scores, and the integration of quality-guided models that improve semantic richness without sacrificing output accuracy.

 
# Hyperparameter Optimization 
### Supervised Learning:
For the baseline model, we experimented with different learning rates and epochs. Through these trials, we observed that a learning rate of **1e-5** resulted in the highest penalized BLEU score within just **7 epochs**. Beyond this point, the loss decreased at a very slow rate, and the penalized BLEU score either stagnated or showed only minimal improvement. On the other hand, using smaller learning rates required more epochs to reach the optimal score, but the training time increased significantly. This insight helped us settle on **1e-5** for the learning rate and **7 epochs** for training.

#### Batch Size:
We selected a **batch size of 8** as it provided a good balance between computational efficiency and model performance. A batch size smaller than 8 led to more variance in the gradient updates, while a larger batch size resulted in higher memory consumption and slower training times. Batch size 8 yielded stable results without sacrificing too much on computational efficiency.

### Hyperparameter Optimization (HPO) for SCST:
For the **SCST** model, some hyperparameters were selected through **Optuna** for automatic tuning, specifically focusing on **rl_lr**, **lambda_rl**, **rl_max_len**, and **rl_batch_size**. The choice of these parameters was made to maximize the penalized BLEU score, which is the task-specific reward.

### Quality-Guided Reward:
The weights for **semantic**, **syntactic**, and **lexical** components were selected based on the empirical results and the trade-off we wanted to achieve between **diversity** and **semantic accuracy**. We experimented with different weight combinations and found that a balanced combination of **0.7** for **semantic similarity**, **0.2** for **syntactic variation**, and **0.1** for **lexical variation** resulted in the most diverse yet semantically coherent paraphrases. This allowed for better control over the model’s output while aligning with the task’s objectives.


# Visualizations 
## **Baseline with padding masking**
<img width="845" height="574" alt="image" src="https://github.com/user-attachments/assets/5bb233bd-77c9-4300-ac88-9eed95583079" />

### Blue curve (Loss):
- The average loss decreases smoothly and consistently as training progresses, indicating stable convergence.
### Red curve (Penalized BLEU):
- The BLEU score improves steadily as the loss decreases, showing a clear correlation between reduced loss and better performance.
### Observation:
- This phase shows a well-behaved supervised training stage, where the model steadily improves without much fluctuation.

## **RL (SCST) on penalized BLEU**
<img width="845" height="574" alt="image" src="https://github.com/user-attachments/assets/96f75088-9f53-45c5-8e6b-f2cecb8a7125" />

### Blue curve (BLEU):
- The BLEU score fluctuates more significantly compared to the first graph. There are ups and downs across epochs.

### Marked point:
- Epoch 10 is highlighted as the best-performing epoch (~31.8 BLEU).

### Observation:
- Reinforcement Learning (RL) training introduces instability in the optimization process, but it occasionally pushes the model to higher performance peaks compared to supervised training.


## **Quality-Guided Reward with Optional Weights**
<img width="845" height="552" alt="image" src="https://github.com/user-attachments/assets/e39b0b90-7ec3-497a-a5aa-26698865e18a" />

### Key Observations:
- The BLEU score shows moderate fluctuations with no clear upward or downward trend.
- The highest BLEU (~29.7) is at epoch 8.
- The score drops sharply to ~21 by epoch 12, suggesting potential overfitting or model degradation.

### Conclusion:

- **Supervised Training**: The graph of supervised training demonstrates a strong and stable base, with a gradual increase in performance. However, it plateaus at a lower BLEU score, suggesting the model achieves its best performance early on but struggles to improve further.

- **RL (SCST) with Penalized BLEU**: The graph of RL (SCST) training shows more fluctuations and instability in performance. While it achieves higher penalized BLEU scores at certain epochs, the training process is less stable. The model occasionally pushes to higher performance peaks, but the fluctuations suggest a less predictable optimization process.

- **Quality-Guided Reward with Optional Weights**: The graph of quality-guided training indicates a significant improvement in performance over the baseline, yet it exhibits the least stability. While the model performs well in terms of penalized BLEU, it faces larger fluctuations and is the least stable approach. Despite this, RL (SCST) consistently achieves higher penalized BLEU scores, showing it as the more effective strategy for this task.

## References 
- _Paraphrase Generation with Deep Reinforcement Learning [Li, Jiang, Shang et al., 2018]_ https://aclanthology.org/D18-1421/
- _Quality Controlled Paraphrase Generation, [Bandel et al., 2022]_ https://aclanthology.org/2022.acl-long.45/


## Bonus Task: BERT paraphrase type detection (**Not completed**):
### Methodology
We tackle Paraphrase Type Detection on the ETPC dataset using the provided minBERT codebase. ETPC is a multi-label (26 types) sentence-pair task and is notably imbalanced.

#### Baseline:
- The ETPC dataloader (SentencePairDataset) yields pairs of sentences plus a 26-dim multi-hot label vector (one bit per paraphrase type).
- In training loop, ETPC batches go through model.predict_paraphrase_types(...) and are trained with BCEWithLogitsLoss (per-label binary cross-entropy).
- Evaluation for ETPC uses a very strict subset accuracy: a prediction is counted correct only if all 26 labels match after thresholding each logit at 0.5.

Result (reference): Dev ≈ 0.068.

Comment: This was the baseline we aimed to improve in our 2nd submission.

#### What we tried:
**1) Cross-encoder for sentence pairs**
Instead of encoding each sentence independently and concatenating two CLS vectors (bi-encoder), we jointly encode the pair as
`[CLS] s1 [SEP] s2 [SEP]`
so self-attention can compare tokens across sentences.

**2) Token-type (segment) IDs for pair modeling**
The custom bert.py initially set token types to zeros. We added optional token_type_ids to BertModel.forward and passed 0 for [CLS] s1 [SEP] and 1 for s2 [SEP]. This is critical so BERT can distinguish the two segments.

**3) Masked mean pooling instead of pooler for ETPC**
 We aggregate token representations with mask-aware mean pooling (excluding pads), which often outperforms the pooler output on multi-label pair tasks.

**Imbalance-aware optimization**
- pos_weight BCEWithLogitsLoss: computed classwise (#neg/#pos) from the ETPC train labels and used it in BCE to up-weight rare types.

However, none of these improvements seemed to improve the baseline accuracy

### Experiments
### B0 — Provided/Reference baseline (minBERT finetune, bi-encoder, linear head)
What: Each sentence encoded independently, CLS vectors concatenated → linear(26).
Expectation: Reasonable multi-label performance; course docs report Dev acc ≈ 0.253 (±0.004) when finetuned.
Result (reference): Dev ≈ 0.068.
Comment: This is the target baseline we aimed to match/exceed.

### Improvements
#### E1 — Cross-encoder + token-type IDs + masked mean pooling + MLP head
- What: Joint input; pass token_type_ids (0/1); mean-pool tokens; small MLP head.
- Expectation: Big jump vs. bi-encoder; cross-sentence attention should help detect fine-grained edits.
- Change vs. B1: Architectural change of the ETPC head and forward path only.
- result (ours): Dev ≈ 0.059 (no consistent gain); Train ≈ 0.068.
- Discussion: No improvement materialized under our setting. Investigation indicated that without proper token-type IDs, cross-encoder helped little; after adding token-types, gains were still muted—likely because subset accuracy punishes even small classwise threshold mismatches under heavy imbalance.

#### E2 — + pos_weight BCE (per-class #neg/#pos) & bias priors
- What: Compute label base rates on ETPC train; use BCEWithLogitsLoss(pos_weight); initialize/freeze last-layer bias to prior log-odds.
- Expectation: Better recall on rare classes; higher exact-set matches.
- Change vs. E1: Optimization & initialization only.
- Result (ours): Dev ≈ 0.059 (stable), Train ≈ 0.068.
- Discussion: Helped training stability (loss decreased), but subset accuracy unchanged. Likely the model’s per-class probabilities improved without crossing the fixed 0.5 threshold for the right labels simultaneously.

### Results:
| Model / Setting                                  | Encoder | Head / Pooling           | Loss                 | Dev Acc (ETPC)                      |
| ------------------------------------------------ | ------- | ------------------------ | -------------------- | ----------------------------------- |
| **Reference baseline** (reported)                | minBERT | Bi-encoder + Linear      | BCE (finetune)       | **0.253 ± 0.004**                   |
| **Our empirical baseline**                       | minBERT | Bi-encoder + Linear      | BCE                  | **0.059**                           |
| E1: Cross-encoder + token-type + mean-pool + MLP | minBERT | Cross-enc + MLP          | BCE                  | **0.059**                           |
| E2: E1 + pos\_weight BCE + bias priors           | minBERT | Cross-enc + MLP          | **BCE(pos\_weight)** | **0.059**                           |

### Discusssion and Takeways
Discussion & Takeaways
- Subset accuracy is extremely unforgiving for ETPC. Even when loss decreases and classwise scores improve, exact match on 26 labels often stays flat under heavy imbalance.
- The ETPC label distribution is highly skewed (e.g., one class at ~0.65, others ~0.001–0.01). This makes a fixed global threshold of 0.5 particularly unhelpful; per-class thresholds help in theory, but didn’t move the metric reliably for us.
- Cross-encoder + token types is generally the right direction, but data scarcity + long-tail + harsh metric can mask gains unless probabilities become very confident and well-calibrated for the correct full set.
- The reference dev accuracy (0.253) suggests that with either (a) different train/dev splits, (b) different preprocessing (e.g., segment localization features), (c) stronger heads, or (d) more careful hyperparameters/regularization, minBERT can do meaningfully better. In our constrained ETPC-only modifications, we didn’t recapture that number.

### References
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT. https://arxiv.org/abs/1810.04805
- Humeau, S., Shuster, K., Lachaux, M.-A., & Weston, J. (2020). Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. ICLR. https://arxiv.org/abs/1905.01969
- King, G., & Zeng, L. (2001). Logistic Regression in Rare Events Data. Political Analysis. https://gking.harvard.edu/publications/logistic-regression-rare-events-data


## Members Contribution 

*Usman Khanzada:* Implemented the paraphrase generation baseline task using BART and then introduced improvements to enhance its performance, also managed group organization

*Ahmad Raza Khawaja:* Implemented the Paraphrase detection basline task using BART and introduced 2 improvements for further optimization, and also implemented baseline implementation for the BERT paraphrase type detection bonus task.

*Bilal Ahmed:* Implemented Sentiment Analysis (SST) baseline and introduced 2 improvements to optimize performance.

*Member 2:*

*Areesha Jiwani:* Implemented Quora Question Pairs (QQP) Baseline and introduced 5 improvements to the baseline to increase performance.

# AI-Usage Card
Artificial Intelligence (AI) aided the development of this project. Please add a link to AI-Usage card [here](https://ai-cards.org/).

#### Paraphrase Type Generation
[![AI-Usage Card SST](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./documentation/AI_usage_card_Paraphrase_generation.pdf/)

#### Paraphrase Type Detection (ETPC)
[![AI-Usage Card SST](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./documentation/AI_Usage_Card_ETPC.pdf/)

#### Sentiment Analysis (SST)
[![AI-Usage Card SST](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./documentation/AI_Usage_Cards_SST.pdf/)

#### Quora Question Pairs (QQP)
[![AI-Usage Card QQP](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./documentation/AI_Usage_Card_QQP.pdf/)



### Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

The project was modified by [Niklas Bauer](https://github.com/ItsNiklas/) for the 2025 DNLP course at the University of Göttingen.
