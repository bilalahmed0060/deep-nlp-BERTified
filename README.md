# DNLP SS25 Final Project

This is the starting code for the default final project for the Deep Learning for Natural Language Processing course at the University of Göttingen. You can find the handout [here](https://docs.google.com/document/d/1pZiPDbcUVhU9ODeMUI_lXZKQWSsxr7GO/edit?usp=sharing&ouid=112211987267179322743&rtpof=true&sd=true)

In this project, you will implement some important components of the BERT model to better understanding its architecture.
You will then use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection and semantic similarity.

After finishing the BERT implementation, you will have a simple model that simultaneously performs the three tasks.
You will then implement extensions to improve on top of this baseline.

## Setup instructions

* Follow `setup.sh` to properly setup a conda environment and install dependencies.
* There is a detailed description of the code structure in [STRUCTURE.md](./STRUCTURE.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh` (Use `setup_gwdg.sh` if you are using the GWDG clusters).
* Libraries that give you other pre-trained models or embeddings are not allowed (e.g., `transformers`).
* Use this template to create your README file of your repository: <https://github.com/gipplab/dnlp_readme_template>

## Project Description

Please refer to the project description for a through explanation of the project and its parts.

### Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

The project was modified by [Niklas Bauer](https://github.com/ItsNiklas/) for the 2025 DNLP course at the University of Göttingen.

####  5) Bart paraphrase type detection

1) Baseline

Model. facebook/bart-large encoder + linear head; pooled by taking token 0; trained with BCE on probabilities.
Thresholding. Fixed 0.5 per class.
Result (dev, 5 epochs):

Accuracy: 0.907

MCC (mean per class): 0.073

##### What problem are we facing

The core issues in our setting

Negative dominance: Each example has a few positives and many negatives per class → BCE over-weights easy negatives.

Long-tailed labels: Head types dominate learning; tail types are ignored.

Label co-occurrence: Simple re-sampling can oversample head labels that co-occur with rare ones, weakening the intended effect.

Hence we decided to change the loss function mainly to cater for these problems above.

2) What loss functions we decided to go with:
- Asymmetric Loss (ASL) addresses the positive/negative imbalance by down-weighting easy negatives while retaining strong gradients for positives; this consistently improves multi-label classification in the long-tail regime. 

- Distribution-Balanced Loss (DB) explicitly re-balances per-label contributions per instance (accounting for co-occurrence) and applies Negative-Tolerant Regularization to mitigate over-suppression of negatives. EMNLP-21 shows DB is particularly effective for multi-label text with long-tailed distributions. 

- Classical class-reweighting (e.g., class-balanced loss using effective number of samples) is helpful, but newer losses above generally perform better in modern multi-label settings.

Therefore we chose ASL as our primary loss (strong empirical gains on tail labels) and DB Loss as a principled alternative targeting co-occurrence + negative dominance, validated for multi-label NLP.

3) What we changed (beyond the loss)

To make the losses effective and to optimize MCC, we also improved representation, calibration, and sampling:

Masked mean pooling for BART (instead of using token 0): stable sentence pair representation for BART encoders.

Train on logits; apply sigmoid only at eval/test (better numerical stability).

Prior-aware logit adjustment (inference-time): subtract τ·logit(prior) per class so rare labels need less evidence (helps recall on tails). We sweep τ∈{0.25,0.5,0.75,1.0}.

Per-class threshold tuning on dev to directly maximize per-class MCC (grid search per label).

WeightedRandomSampler at instance level: instance weight = sum of inverse class priors of its positive labels, to surface tail-heavy examples without naive per-label oversampling.

4) Results
All runs are 5 epochs on the same train/dev split.

Summary table:

| Method             | Extra tricks                | Dev accuracy | Dev MCC (fixed 0.5) | Dev MCC (threshold-tuned) |
| ------------------ | --------------------------- | -----------: | ------------------: | ------------------------: |
| **Baseline (BCE)** | none                        |        0.907 |           **0.073** |                         — |
| **DB Loss**        | τ=0.5, per-class thresholds |        0.762 |               0.148 |                 **0.296** |
| **ASL**            | τ=0.5, per-class thresholds |        0.431 |               0.139 |                 **0.434** |


- Notes:
• “Fixed 0.5” uses a single threshold for every label. “Threshold-tuned” picks a threshold per label to maximize dev MCC (after optional logit adjustment).
• For DB, raw epoch-5 MCC (fixed 0.5) rose from ~0.000 to 0.209 during training; with calibration it reached 0.296.
• For ASL, raw epoch-5 MCC reached 0.340; with calibration it reached 0.434.

5) Takeaways:
4.2 Takeaways

Loss matters: Replacing BCE with ASL or DB increases tail recall and MCC markedly.

Calibration matters even more: Per-class thresholds + τ-adjustment roughly doubled MCC vs fixed 0.5.

ASL > DB on our data: ASL delivered the best tuned MCC (0.434). This aligns with reports that ASL excels when negative dominance is the main bottleneck. 

DB is competitive and theoretically appealing when label co-occurrence is severe; EMNLP-21 shows DB’s advantages for multi-label text, which we partially observe (MCC 0.296 vs baseline 0.073).

6) References
Asymmetric Loss (ASL): Ridnik et al., Asymmetric Loss for Multi-Label Classification, ICCV 2021. [CVF open access](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf)

Distribution-Balanced Loss (DB): Wu et al., Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets, ECCV 2020 (and arXiv). [ECVA](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490154.pdf), [arXiv](https://arxiv.org/abs/2007.09654)

DB for multi-label text: Huang et al., Balancing Methods for Multi-Label Text Classification with Long-Tailed Class Distribution, EMNLP 2021. [ACL Anthology](https://aclanthology.org/2021.emnlp-main.643/)

Class-Balanced Loss (background): Cui et al., CVPR 2019. [CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)