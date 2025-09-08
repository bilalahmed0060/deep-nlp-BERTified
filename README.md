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
##### ** 4.1 Prior-aware Logit Adjustment**
We adjust logits at inference (and optionally validation) by subtracting τ * logit(prior) where prior = P(y=1) from training data.

- Effect: Down-weights frequent classes and up-weights rare ones without touching learned weights.
- Setting: We used τ=0.5 in our best configurations.

##### ** 4.2 Prior-aware Logit Adjustment**
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

### Vislualisation
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

## 5) Paraphrase generation using BART:
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


## Members Contribution 

*Usman Khanzada:* Implemented the paraphrase generation baseline task using BART and then introduced improvements to enhance its performance, also managed group organization

*Ahmad Raza Khawaja:* Implemented the Paraphrase detection basline task using BART and introduced 2 improvements for further optimization, and also implemented baseline implementation for the BERT paraphrase type detection bonus task.

*Member 2:*

*Member 2:*

*Member 2:*

# AI-Usage Card
Artificial Intelligence (AI) aided the development of this project. Please add a link to your AI-Usage card [here](https://ai-cards.org/).

### Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

The project was modified by [Niklas Bauer](https://github.com/ItsNiklas/) for the 2025 DNLP course at the University of Göttingen.
