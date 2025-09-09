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

*Member 2:*

# AI-Usage Card
Artificial Intelligence (AI) aided the development of this project. Please add a link to AI-Usage card [here](https://ai-cards.org/).

#### Paraphrase Type Generation
[![AI-Usage Card SST](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./documentation/AI_usage_card_Paraphrase_generation.pdf/)

#### Paraphrase Type Detection (ETPC)
[![AI-Usage Card SST](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./documentation/AI_Usage_Card_ETPC.pdf/)

#### Sentiment Analysis (SST)
[![AI-Usage Card SST](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./documentation/AI_Usage_Cards_SST.pdf/)


### Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

The project was modified by [Niklas Bauer](https://github.com/ItsNiklas/) for the 2025 DNLP course at the University of Göttingen.
