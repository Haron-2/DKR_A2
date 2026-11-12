# Heterophily: When Neighbors Mislead

## A controlled comparison of MLP, GCN, GAT and H2GCN on Chameleon, Squirrel and Actor


**Course:** Data and Knowledge Representation — Assignment A2

**Team:** 9

**Members:** Ilya Ponomarev, Maksim Kirillov

**Code:** `https://github.com/Haron-2/DKR_A2`

**Artifacts:** `results/`, `reports/figures/`, `reports/tables/`


---


## Abstract

We compare a feature-only MLP with GCN, GAT and H2GCN on three heterophilous benchmarks—Chameleon, Squirrel and Actor—using original and filtered data. The evaluation covers 240 primary runs and 80 H2GCN ablations across ten splits, with paired comparisons and Holm correction.

On original Chameleon and Squirrel, all graph models outperform the MLP by 8.8–13.1 accuracy points. On the filtered versions, most gains disappear; only H2GCN on filtered Squirrel retains a significant improvement (+3.15 pp). On Actor, GCN and GAT underperform the MLP by 5.4–6.1 pp, while H2GCN matches its accuracy but has lower macro-F1. Feature-based auditing finds 4.2–22.3% mean train–test overlap, but low label agreement prevents interpreting this as confirmed label leakage. Ablations identify the one-hop branch, rather than exact two-hop aggregation, as H2GCN's main useful component. Since filtering also changes node populations and splits, the results do not isolate leakage as the cause.

---


## 1. Introduction

Heterophily occurs when graph edges tend to connect nodes from different classes. In a conventional message-passing layer, each node combines its own representation with information aggregated from its neighbours. When neighbouring nodes usually belong to other classes, this aggregation can dilute useful class information and hurt classification.

H2GCN (Zhu et al., 2020) is designed for this setting. It keeps a node's own representation separate from its neighbours' representations, uses both one-hop and higher-order neighbourhoods, and concatenates intermediate representations rather than repeatedly replacing them through stacked layers.

This study addresses two evaluation concerns:

1. **A feature-only baseline is essential.** Without an MLP that ignores graph edges, we cannot tell whether a GNN benefits from graph structure or simply from node features. We train the MLP under the same protocol and define the per-split **Structural Delta** as
   `Δ = Accuracy(model) − Accuracy(MLP)`.
2. **Benchmark duplicates may inflate reported performance.** Platonov et al. (2023) identify duplicate nodes in Chameleon and Squirrel that can appear across train, validation and test sets. Their filtered datasets and revised splits are intended to address this issue. We therefore evaluate each model on both the original (`raw`) and filtered (`filtered`) versions.

We ask five questions:

- **RQ1:** Does graph structure improve accuracy over the MLP baseline?
- **RQ2:** When does GCN help or hurt, and does attention (GAT) change that behaviour?
- **RQ3:** Does H2GCN benefit from exact two-hop neighbourhoods?
- **RQ4:** How do the conclusions change on the filtered datasets?
- **RQ5:** How much feature-identical overlap remains between train and test nodes, and are the performance differences statistically and practically meaningful?


---


## 2. Datasets

| Condition | Nodes | Features | Classes | Undirected edges | Average degree | Isolated | Edge homophily | Node homophily | Adjusted homophily | Adjusted (node prop.) |
|---|---|---|---|---|---|---|---|---|---|---|
| Chameleon / raw | 2277 | 2325 | 5 | 31 371 | 27.55 | 0 | 0.2299 | 0.2471 | 0.0324 | 0.0353 |
| Chameleon / filtered | 890 | 2325 | 5 | 8 854 | 19.90 | 0 | 0.2361 | 0.2441 | 0.0295 | 0.0320 |
| Squirrel / raw | 5201 | 2089 | 5 | 198 353 | 76.27 | 0 | 0.2221 | 0.2172 | 0.0067 | 0.0277 |
| Squirrel / filtered | 2223 | 2089 | 5 | 46 998 | 42.28 | 0 | 0.2072 | 0.1905 | 0.0086 | **−0.0340** |
| Actor / raw | 7600 | 932 | 5 | 26 659 | 7.02 | 0 | 0.2167 | 0.2199 | 0.0028 | 0.0042 |
| Actor / filtered | 7600 | 932 | 5 | 26 659 | 7.02 | 0 | 0.2167 | 0.2199 | 0.0028 | 0.0042 |


*Dataset statistics from `scripts/analyze_graphs.py`

(`reports/dataset_summary.csv`). Edges are counted once (undirected coalesced

graph); average degree is `2|E|/N`. "Adjusted" uses degree-weighted class

proportions (Lim et al., 2021), "Adjusted (node prop.)" uses plain class

proportions `p_k = n_k/N`. All three graphs are strongly heterophilous: edge

homophily ≈ 0.21–0.24. Adjusted homophily is close to zero everywhere and is

**negative** for Squirrel filtered (−0.034 under node proportions), i.e. edges are

less homophilous than random node pairs; the measure is not bounded below by zero.*


**Class balance differs substantially between conditions** (`class_counts` in

`reports/dataset_summary.csv`):

| Condition | Class counts (classes 0–4) | Max / min |
|---|---|---|
| Chameleon / raw | 456, 460, 453, 521, 387 | 1.35 |
| Chameleon / filtered | 242, 134, 209, 164, 141 | 1.81 |
| Squirrel / raw | 1042, 1040, 1039, 1040, 1040 | 1.00 |
| Squirrel / filtered | 756, 516, 397, 321, 233 | 3.24 |
| Actor / raw | 853, 1337, 1630, 1815, 1965 | 2.30 |
| Actor / filtered | 853, 1337, 1630, 1815, 1965 | 2.30 |


Chameleon raw is close to balanced (max/min = 1.35), and Squirrel raw is almost perfectly balanced (1.00). Squirrel filtered (3.24) and Actor (2.30) are more imbalanced. Filtering therefore changes class balance as well as graph size. We report macro-F1 and per-class results alongside accuracy because accuracy alone can hide poor performance on minority classes.

**Origin and splits.**

- **Chameleon and Squirrel** are Wikipedia page networks (Rozemberczki et al., 2021). Their nodes use bag-of-words features and belong to five traffic classes defined by Pei et al. (2020). The `raw` condition uses the Geom-GCN files and their ten `0.6/0.2/0.2` train/validation/test splits (pinned code revision).
- **Filtered Chameleon and Squirrel** come from the Platonov et al. (2023) redistribution (`chameleon_filtered.npz` and `squirrel_filtered.npz`, pinned revision). These files remove nodes matching the authors' duplicate criterion—no incoming edges, with the same target and outgoing edge set—and provide ten revised masks. Our node counts (890 and 2,223) match their reported non-duplicate counts, and our class counts match their Table 5.
- **Actor** is the film co-occurrence network from Pei et al. (2020), using the Geom-GCN `film` splits in the `raw` condition. No upstream duplicate-filtered Actor dataset is available, and the distributed `actor.npz` reproduces the original masks. We therefore label `actor/filtered` as a **custom reproduction**: the graph and labels are unchanged, but we generate ten class-stratified 50/25/25 splits using seed `20230222`. This distinction is recorded in the dataset metadata and retained throughout the report.

The raw and filtered Chameleon/Squirrel datasets contain different nodes. Actor uses the same graph in both conditions, but its split masks differ. We therefore do not pair raw-versus-filtered results for any dataset; those comparisons are descriptive only.


---


## 3. Data audit: duplicates and train-test leakage


**Method.**

We group nodes whose feature vectors are identical or nearly identical, using the tolerance `||x_i − x_j||₂ < 1e-6`. Exact duplicate rows are collapsed first; for non-integral features, near-duplicates are identified with a blocked Euclidean computation. This avoids constructing a full `N × N` distance matrix. A duplicate group is a maximal set of nodes connected by this tolerance relation.

For split `s`, **train–test feature overlap** is the share of test nodes whose duplicate group contains at least one training node. This measures feature reuse, not label leakage: matching features do not guarantee matching labels. The audit is performed independently of model training, and no duplicate-group information is given to the models.

| Condition | Duplicate groups | Duplicate pairs | Nodes in duplicate groups | Largest group | Groups with conflicting labels | Label agreement | Mean feature overlap | Overlap range |
|---|---|---|---|---|---|---|---|---|
| Chameleon / raw | 48 | 27 264 | 375 (16.5 %) | 233 | 20 | 19.7 % | 14.45 % | 12.94–17.32 % |
| Chameleon / filtered | 5 | 4 384 | 105 (11.8 %) | 94 | 2 | 8.6 % | 10.81 % | 8.99–15.70 % |
| Squirrel / raw | 44 | 13 608 | 265 (5.1 %) | 165 | 27 | 13.2 % | 4.21 % | 3.55–5.09 % |
| Squirrel / filtered | 20 | 6 364 | 158 (7.1 %) | 113 | 8 | 15.2 % | 5.83 % | 4.44–6.63 % |
| Actor / raw | 390 | 31 362 | 1976 (26.0 %) | 172 | 333 | 6.2 % | 22.34 % | 19.87–23.49 % |
| Actor / filtered | 390 | 31 362 | 1976 (26.0 %) | 172 | 333 | 6.2 % | 22.07 % | 19.83–23.67 % |


*Audit produced by `scripts/audit_duplicates.py`

(`results/analysis/duplicate_audit.json`, per-split values in the `.csv`).*


**Four different quantities, which must not be conflated.**

The audit involves four distinct concepts:

1. **Platonov et al.'s duplicate nodes.** Their criterion combines graph structure and targets: a node has no incoming edges and shares both its regression target and outgoing edge set with another node. Removing such nodes produces the 890-node Chameleon and 2,223-node Squirrel datasets used here. We use the released filtered files; we do not independently reproduce or verify their node-removal procedure because the original regression targets are not included in the classification data.
2. **Feature-identical groups in our audit.** These groups are defined only by `||x_i − x_j||₂ < 1e-6`. They do not take neighbourhoods or targets into account. They contain 5.1% of nodes in raw Squirrel and up to 26.0% in Actor. Even after upstream filtering, they still contain 11.8% of Chameleon nodes and 7.1% of Squirrel nodes.
3. **Train–test feature overlap.** This is the share of test nodes with a feature-identical counterpart in the training split: 4.21% in raw Squirrel and up to 22.34% in Actor. The filtered Chameleon and Squirrel datasets still have 10.81% and 5.83% overlap, respectively. Such overlap is necessary for direct feature-based label copying, but it is not sufficient.
4. **Label agreement within duplicate groups.** Only 6.2%–19.7% of duplicate nodes belong to groups whose labels are all identical; between 2 and 333 groups per condition contain conflicting labels. Thus, two nodes can have the same features but different labels.

We therefore describe the measured quantity as **feature overlap**, not as proven label leakage. We cannot claim that every test node with a training feature twin could have its label copied. The narrower finding is that all three datasets contain non-trivial train–test feature overlap, and the upstream filtering of Chameleon and Squirrel does not eliminate it.


![Duplicate and leakage audit](figures/duplicate_leakage.png)


---


## 4. Methods


All four models expose the same interface: `forward(x, edge_index) → logits` and `get_embeddings(x, edge_index) → h`. The embedding `h` is the exact tensor passed to the final classifier. Every model uses dropout 0.5, and each output has shape `[N, C]`, where `C = 5`.

### 4.1 MLP (structure-blind baseline)


```

h = Dropout(ReLU(LayerNorm(W1 x + b1))),   logits = W2 h + b2

```


The MLP uses layer normalisation rather than batch normalisation, so full-graph inference does not depend on batch statistics. It accepts `edge_index` for interface compatibility but never uses it; a unit test confirms that changing or removing the edge list leaves its logits unchanged. The hidden and embedding widths are both 64.

### 4.2 GCN


```

h = Dropout(ReLU(GCNConv1(x, A + I))),   logits = GCNConv2(h, A + I)

```


Each `GCNConv` layer adds self-loops and applies the symmetric normalisation `D^{-1/2}(A+I)D^{-1/2}`. The model has two layers and a hidden width of 64.

### 4.3 GAT


```

h = Dropout(ELU(GATConv1(x, heads=8, out_channels=8))),  logits = GATConv2(h, heads=1, concat=False)

```


The hidden layer concatenates eight attention heads, each with eight channels, for 64 features in total. The output layer uses a single head with `concat=False`, so the logits contain exactly `C` class scores rather than `C × heads` values.

### 4.4 H2GCN (core principles)


For a projected input `H0 = ReLU(X W0)`:


```

A1 = row-normalised adjacency without self-loops

A2 = row-normalised binarised exact two-hop adjacency

     A2 = (A² > 0) ∧ (A = 0) ∧ (I = 0)


R1(1) = A1 H0,  R2(1) = A2 H0,  H1 = [H0, R1(1), R2(1)]

R1(2) = A1 H1,  R2(2) = A2 H1

H_final = [H0, R1(1), R2(1), R1(2), R2(2)]

logits  = Dropout(H_final) W_c + b_c

```


- **Exact two-hop adjacency.** We construct `A2` directly rather than approximating it with `A1(A1H) − A1H − H`. The sparse product is computed in row blocks to avoid materialising the full matrix at once. We then remove self-connections and one-hop pairs and row-normalise the remaining entries. Duplicate edges, isolated nodes and empty two-hop neighbourhoods are handled explicitly and covered by unit tests.
- **Trainable transformations.** The diffusion branches contain no trainable transformations. The only learned components are the input projection `W0` and the output classifier; dropout is applied once to the final concatenation.
- **Embedding width.** The full model produces a 576-dimensional embedding (`9 × 64`): three 64-dimensional blocks after the first round and six after the second. We retain this width rather than forcing it to 64, because doing so would change the architecture being evaluated.
- **Dense computation when appropriate.** If the exact two-hop adjacency contains more than 5% of all `N²` possible pairs (8.4 million of 27 million for raw Squirrel), we store it densely and use BLAS. This changes the computation strategy, not the underlying operator.

Our model follows the core **H2GCN principles**, but it is not a bit-exact reproduction of the authors' implementation, which we could not retrieve. The main differences are row-normalised diffusion, one dropout operation on the final concatenation and no per-layer batch normalisation; these are documented in the module docstring.

### 4.5 Parameter counts and capacity

| Dataset | MLP | GCN | GAT | H2GCN (full) | H2GCN (ablation) |
|---|---|---|---|---|---|
| Chameleon | 149 317 | 149 189 | 149 327 | 151 749 | 150 149 |
| Squirrel | 134 213 | 134 085 | 134 223 | 136 645 | 135 045 |
| Actor | 60 165 | 60 037 | 60 175 | 62 597 | not run |


*Parameter counts come from `models.registry.count_parameters`. H2GCN has only an input projection and an output classifier, so its parameter count is close to those of the other models despite its nine-times-wider embedding. Its results therefore cannot be explained by a substantially larger number of learned parameters.*

---


## 5. Experimental protocol

- **Experiment grid.** The primary grid covers 4 models × 3 datasets × 2 data conditions × 10 splits = 240 runs. Two H2GCN ablation variants are evaluated on Chameleon and Squirrel, in both conditions and across ten splits, adding 80 runs. Total: 320 runs.
- **Training.** All models use Adam (learning rate 0.01, weight decay `5e-4`), cross-entropy loss and dropout 0.5. Training is full-batch and transductive. Early stopping monitors validation loss (`patience = 50`, `min_delta = 1e-4`), with a maximum of 1,000 epochs. Before testing, we restore the checkpoint with the best validation loss.
- **Data masks.** For split `s`, the masks are `train_mask[:, s]`, `val_mask[:, s]` and `test_mask[:, s]`. Only training labels contribute to the loss; validation labels control early stopping; test labels are used once, after checkpoint restoration.
- **Seeds.** Split `s` uses seed `42 + s` for all four models. This pairs model comparisons by split. Every run starts from a fresh model and writes a separate checkpoint.
- **Metrics.** We record test accuracy and macro-F1, best validation loss, best epoch, epochs trained, parameter count, and training/evaluation time.
- **Statistical tests.** For each model, we compare per-split results with the MLP using paired t-tests and Wilcoxon signed-rank tests. We report 95% confidence intervals for the mean paired difference and Cohen's `d_z`. Holm correction is applied across all 18 primary comparisons for each metric (six conditions × three graph models). Raw and filtered conditions are not paired.
- **Software and hardware.** Python 3.12, `torch==2.8.0+cpu`, `torch-geometric==2.7.0`, `numpy==2.3.4`, `scipy==1.18.1`, `scikit-learn==1.9.1` and `matplotlib==3.11.2`. Experiments ran on a CPU-only machine with four physical cores, 15.3 GB RAM and eight PyTorch intra-op threads.
- **Deviations.** `actor/filtered` uses the custom stratified splits described above. H2GCN follows the core principles rather than the exact published implementation. No hyperparameters were selected using test performance; `configs/full.json` fixes the same protocol for every model.


---


## 6. Main results


All 240 primary runs completed successfully, with no failures or missing results. Each cell below summarises ten runs.

### 6.1 Accuracy (mean ± SD over ten splits)

| Condition | MLP | GCN | GAT | H2GCN |
|---|---|---|---|---|
| Chameleon / raw | 0.5009 ± 0.0193 | **0.6246 ± 0.0182** | **0.6314 ± 0.0320** | 0.6048 ± 0.0287 |
| Chameleon / filtered | 0.3757 ± 0.0463 | 0.3592 ± 0.0666 | 0.3875 ± 0.0357 | **0.4133 ± 0.0353** |
| Squirrel / raw | 0.3329 ± 0.0099 | **0.4626 ± 0.0220** | 0.4316 ± 0.0133 | 0.4212 ± 0.0163 |
| Squirrel / filtered | 0.3474 ± 0.0148 | 0.3468 ± 0.0171 | 0.3442 ± 0.0156 | **0.3789 ± 0.0171** |
| Actor / raw | 0.3514 ± 0.0090 | 0.2978 ± 0.0097 | 0.2936 ± 0.0106 | **0.3570 ± 0.0077** |
| Actor / filtered | 0.3577 ± 0.0094 | 0.3007 ± 0.0065 | 0.2964 ± 0.0097 | **0.3633 ± 0.0083** |


### 6.2 Macro-F1 (mean ± SD over ten splits)

| Condition | MLP | GCN | GAT | H2GCN |
|---|---|---|---|---|
| Chameleon / raw | 0.4913 ± 0.0204 | 0.6249 ± 0.0183 | **0.6320 ± 0.0320** | 0.6057 ± 0.0298 |
| Chameleon / filtered | 0.3033 ± 0.0462 | 0.2834 ± 0.0741 | 0.3165 ± 0.0253 | **0.3519 ± 0.0416** |
| Squirrel / raw | 0.3136 ± 0.0142 | **0.4536 ± 0.0232** | 0.4232 ± 0.0143 | 0.4044 ± 0.0232 |
| Squirrel / filtered | 0.2295 ± 0.0307 | 0.2241 ± 0.0228 | 0.1889 ± 0.0127 | **0.2624 ± 0.0219** |
| Actor / raw | **0.3190 ± 0.0152** | 0.2110 ± 0.0135 | 0.2011 ± 0.0129 | 0.2952 ± 0.0108 |
| Actor / filtered | **0.3266 ± 0.0149** | 0.2095 ± 0.0096 | 0.2056 ± 0.0059 | 0.3000 ± 0.0084 |


![Accuracy by model and condition](figures/accuracy_comparison.png)


![Macro-F1 by model and condition](figures/macro_f1_comparison.png)


### 6.3 Structural Delta (Accuracy − MLP, paired per split)

| Condition | GCN | GAT | H2GCN |
|---|---|---|---|
| Chameleon / raw | +12.37 ± 2.73 | +13.05 ± 3.61 | +10.39 ± 2.86 |
| Chameleon / filtered | −1.65 ± 7.91 | +1.18 ± 3.10 | +3.76 ± 4.14 |
| Squirrel / raw | +12.98 ± 2.28 | +9.88 ± 1.48 | +8.84 ± 1.89 |
| Squirrel / filtered | −0.07 ± 2.00 | −0.32 ± 1.78 | +3.15 ± 1.98 |
| Actor / raw | −5.36 ± 1.33 | −5.78 ± 1.21 | +0.55 ± 1.17 |
| Actor / filtered | −5.70 ± 1.17 | −6.13 ± 1.44 | +0.56 ± 1.29 |


*All values are percentage points of accuracy. The Structural Delta is computed on

matching splits and the same seed, so positive values mean that the graph model

beats the feature-only baseline on the same data.*


![Structural Delta](figures/structural_delta_accuracy.png)


### 6.4 Raw versus filtered

| Dataset | Model | Raw | Filtered | Δ (filtered − raw) |
|---|---|---|---|---|
| Chameleon | MLP | 0.5009 | 0.3757 | −12.52 |
| Chameleon | GCN | 0.6246 | 0.3592 | −26.54 |
| Chameleon | GAT | 0.6314 | 0.3875 | −24.39 |
| Chameleon | H2GCN | 0.6048 | 0.4133 | −19.15 |
| Squirrel | MLP | 0.3329 | 0.3474 | +1.46 |
| Squirrel | GCN | 0.4626 | 0.3468 | −11.59 |
| Squirrel | GAT | 0.4316 | 0.3442 | −8.74 |
| Squirrel | H2GCN | 0.4212 | 0.3789 | −4.23 |
| Actor | MLP | 0.3514 | 0.3577 | +0.63 |
| Actor | GCN | 0.2978 | 0.3007 | +0.29 |
| Actor | GAT | 0.2936 | 0.2964 | +0.28 |
| Actor | H2GCN | 0.3570 | 0.3633 | +0.64 |


*Descriptive only: the node populations differ for Chameleon/Squirrel and the

split masks differ for Actor, so no paired test is computed.*


![Raw versus filtered](figures/raw_vs_filtered_accuracy.png)


**Reading the results.**

The results show that the value of message passing depends strongly on the dataset and data condition:

- **Original Chameleon and Squirrel:** GCN, GAT and H2GCN all outperform the MLP. Gains are about 12–13 pp on Chameleon and 9–13 pp on Squirrel.
- **Filtered Chameleon and Squirrel:** the advantage mostly disappears. GCN is 1.65 pp below the MLP on Chameleon and effectively tied on Squirrel. GAT is also statistically indistinguishable from the MLP on both datasets. H2GCN retains the clearest advantage, particularly on filtered Squirrel (+3.15 pp).
- **Actor:** GCN and GAT are significantly below the MLP by 5.4–6.1 pp. H2GCN is statistically tied with the MLP in accuracy. This suggests that, on Actor, neighbourhood aggregation can dilute information that is already present in node features.

The best-performing architecture also changes across conditions: GAT leads on the original Chameleon and Squirrel splits, whereas H2GCN leads on the filtered versions and on Actor. GCN is not the best model in any filtered condition.


### 6.5 Training dynamics and cost


Early stopping behaves differently across models and data conditions. The MLP typically stops early (best epoch 2–7; 52–57 epochs trained). On the original data, GCN trains longer: 125 epochs on average for Chameleon and 170 for Squirrel, with best epochs of 75 and 120. On filtered data, all models stop after 52–61 epochs. Per-epoch losses for every run are available in `results/logs/<run_id>.csv`; the figure below shows split 0 on raw Chameleon.


![Training convergence](figures/training_convergence_chameleon_raw.png)


The 240 primary runs required 2.12 hours of accumulated training time; the 80 ablations required another 0.26 hours. Total wall-clock time was approximately 3.5 hours because runs were executed sequentially and the wall-clock measurement includes data loading and operator construction. H2GCN on Actor was the most expensive configuration (about 215 seconds per run): its exact two-hop operator covers a large fraction of all node pairs and is evaluated densely.


### 6.6 Per-class behaviour and confusion matrices


**Per-class diagnostics.**

Accuracy and macro-F1 tell different stories in two conditions. To understand why, we re-evaluated all saved best checkpoints in inference mode, without retraining, using `scripts/evaluate_predictions.py`. Recomputed test accuracy matches the recorded value for all 240 runs (maximum absolute difference: 0.0), confirming that the saved checkpoints and masks reproduce the original evaluation. The resulting confusion matrices and per-class metrics are stored in `results/analysis/confusion_matrices.json`, `per_class_metrics.csv` and `per_class_summary.csv`.


**Actor raw** (mean test support per class: 168.5, 265.6, 333.7, 368.5, 383.7;

macro-F1 in parentheses):

| Model | F1 c0 | F1 c1 | F1 c2 | F1 c3 | F1 c4 | Predicted share c0…c4 | Macro-F1 |
|---|---|---|---|---|---|---|---|
| MLP | 0.222 | 0.275 | 0.235 | 0.291 | 0.573 | 0.093, 0.169, 0.189, 0.247, 0.302 | 0.319 |
| GCN | 0.000 | 0.071 | 0.230 | 0.300 | 0.454 | 0.000, 0.032, 0.219, 0.335, 0.414 | 0.211 |
| GAT | 0.000 | 0.054 | 0.203 | 0.309 | 0.439 | 0.000, 0.023, 0.168, 0.364, 0.445 | 0.201 |
| H2GCN | 0.014 | 0.239 | 0.311 | 0.326 | 0.585 | 0.002, 0.145, 0.345, 0.262, 0.245 | 0.295 |


**Squirrel filtered** (mean test support: 157.0, 100.6, 78.7, 64.7, 46.7):

| Model | F1 c0 | F1 c1 | F1 c2 | F1 c3 | F1 c4 | Predicted share c0…c4 | Macro-F1 |
|---|---|---|---|---|---|---|---|
| MLP | 0.551 | 0.215 | 0.148 | 0.092 | 0.142 | 0.563, 0.239, 0.107, 0.058, 0.033 | 0.230 |
| GCN | 0.545 | 0.184 | 0.102 | 0.115 | 0.175 | 0.640, 0.160, 0.066, 0.066, 0.069 | 0.224 |
| GAT | 0.537 | 0.122 | 0.105 | 0.106 | 0.074 | 0.745, 0.108, 0.069, 0.056, 0.022 | 0.189 |
| H2GCN | 0.586 | 0.172 | 0.179 | 0.185 | 0.190 | 0.617, 0.134, 0.131, 0.086, 0.033 | 0.262 |


**Measured facts.***

The per-class results explain the metric differences:

1. **Actor raw:** GCN and GAT never predict the smallest class (mean test support 168.5; F1 = 0.000), while the MLP reaches F1 = 0.222. H2GCN predicts that class only 0.2% of the time (F1 = 0.014). This minority-class failure explains why H2GCN is close to the MLP in accuracy but lower in macro-F1, and why GCN/GAT fall 10.8–12.1 F1 points behind.
2. **Squirrel filtered:** GAT assigns 74.5% of its test predictions to the majority class. Its F1 on the four minority classes is only 0.074–0.122, so accuracy (0.344) hides a pronounced imbalance that macro-F1 (0.189) reveals.
3. **Squirrel filtered:** H2GCN has the highest F1 on each minority class (0.172–0.190), which accounts for its stronger macro-F1 in this condition.

These patterns come from the saved predictions, not from interpreting accuracy alone. Section 10 discusses possible explanations.


---


## 7. Statistical analysis


Every comparison pairs the model with the MLP on the **same** dataset, condition and

split (n = 10 pairs), never across conditions. Confidence intervals are 95 % t

intervals of the mean paired difference, `d_z` is Cohen's effect size for paired

data, and the Holm correction is applied across the family of all 18 tests of a

metric (six conditions × three model comparisons). "Significant" always refers to

the Holm-corrected p-value at α = 0.05.


### 7.1 Accuracy

| Condition | Comparison | Mean Δ (pp) | 95 % CI (pp) | t | p (raw) | p (Holm) | p (Wilcoxon) | d_z | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Chameleon / raw | GCN | +12.37 | [+10.41, +14.32] | 14.32 | < 0.0001 | < 0.0001 | 0.0020 | 4.53 | significant |
| Chameleon / raw | GAT | +13.05 | [+10.46, +15.63] | 11.42 | < 0.0001 | < 0.0001 | 0.0020 | 3.61 | significant |
| Chameleon / raw | H2GCN | +10.39 | [+8.35, +12.44] | 11.49 | < 0.0001 | < 0.0001 | 0.0020 | 3.63 | significant |
| Chameleon / filtered | GCN | −1.65 | [−7.31, +4.00] | −0.66 | 0.525 | 1.000 | 0.625 | −0.21 | not significant |
| Chameleon / filtered | GAT | +1.18 | [−1.04, +3.39] | 1.20 | 0.261 | 1.000 | 0.313 | 0.38 | not significant |
| Chameleon / filtered | H2GCN | +3.76 | [+0.80, +6.72] | 2.87 | 0.018 | 0.129 | 0.027 | 0.91 | not significant |
| Squirrel / raw | GCN | +12.98 | [+11.35, +14.61] | 18.00 | < 0.0001 | < 0.0001 | 0.0020 | 5.69 | significant |
| Squirrel / raw | GAT | +9.88 | [+8.82, +10.94] | 21.07 | < 0.0001 | < 0.0001 | 0.0020 | 6.66 | significant |
| Squirrel / raw | H2GCN | +8.84 | [+7.48, +10.19] | 14.76 | < 0.0001 | < 0.0001 | 0.0020 | 4.67 | significant |
| Squirrel / filtered | GCN | −0.07 | [−1.50, +1.36] | −0.11 | 0.918 | 1.000 | 0.922 | −0.03 | not significant |
| Squirrel / filtered | GAT | −0.32 | [−1.60, +0.96] | −0.57 | 0.583 | 1.000 | 0.557 | −0.18 | not significant |
| Squirrel / filtered | H2GCN | +3.15 | [+1.73, +4.57] | 5.02 | 0.0007 | 0.0058 | 0.0020 | 1.59 | significant |
| Actor / raw | GCN | −5.36 | [−6.31, −4.41] | −12.74 | < 0.0001 | < 0.0001 | 0.0020 | −4.03 | significant (worse) |
| Actor / raw | GAT | −5.78 | [−6.65, −4.92] | −15.15 | < 0.0001 | < 0.0001 | 0.0020 | −4.79 | significant (worse) |
| Actor / raw | H2GCN | +0.55 | [−0.28, +1.39] | 1.50 | 0.168 | 1.000 | 0.186 | 0.47 | not significant |
| Actor / filtered | GCN | −5.70 | [−6.53, −4.86] | −15.44 | < 0.0001 | < 0.0001 | 0.0020 | −4.88 | significant (worse) |
| Actor / filtered | GAT | −6.13 | [−7.16, −5.10] | −13.48 | < 0.0001 | < 0.0001 | 0.0020 | −4.26 | significant (worse) |
| Actor / filtered | H2GCN | +0.56 | [−0.36, +1.49] | 1.38 | 0.202 | 1.000 | 0.289 | 0.44 | not significant |


### 7.2 Macro-F1

| Condition | Comparison | Mean Δ (pp) | p (raw) | p (Holm) | p (Wilcoxon) | d_z | Verdict |
|---|---|---|---|---|---|---|---|
| Chameleon / raw | GCN | +13.36 | < 0.0001 | < 0.0001 | 0.0020 | 4.58 | significant |
| Chameleon / raw | GAT | +14.07 | < 0.0001 | < 0.0001 | 0.0020 | 3.79 | significant |
| Chameleon / raw | H2GCN | +11.44 | < 0.0001 | < 0.0001 | 0.0020 | 3.88 | significant |
| Chameleon / filtered | GCN | −2.00 | 0.481 | 0.962 | 0.625 | −0.23 | not significant |
| Chameleon / filtered | GAT | +1.32 | 0.186 | 0.558 | 0.193 | 0.45 | not significant |
| Chameleon / filtered | H2GCN | +4.86 | 0.016 | 0.078 | 0.027 | 0.94 | not significant |
| Squirrel / raw | GCN | +14.00 | < 0.0001 | < 0.0001 | 0.0020 | 7.70 | significant |
| Squirrel / raw | GAT | +10.97 | < 0.0001 | < 0.0001 | 0.0020 | 6.53 | significant |
| Squirrel / raw | H2GCN | +9.09 | < 0.0001 | < 0.0001 | 0.0020 | 3.77 | significant |
| Squirrel / filtered | GCN | −0.54 | 0.686 | 0.962 | 0.922 | −0.13 | not significant |
| Squirrel / filtered | GAT | −4.06 | 0.002 | 0.014 | 0.004 | −1.33 | significant (worse) |
| Squirrel / filtered | H2GCN | +3.29 | 0.026 | 0.102 | 0.027 | 0.85 | not significant |
| Actor / raw | GCN | −10.80 | < 0.0001 | < 0.0001 | 0.0020 | −7.45 | significant (worse) |
| Actor / raw | GAT | −11.79 | < 0.0001 | < 0.0001 | 0.0020 | −7.35 | significant (worse) |
| Actor / raw | H2GCN | −2.38 | < 0.0001 | 0.0001 | 0.0020 | −2.59 | significant (worse) |
| Actor / filtered | GCN | −11.71 | < 0.0001 | < 0.0001 | 0.0020 | −7.77 | significant (worse) |
| Actor / filtered | GAT | −12.09 | < 0.0001 | < 0.0001 | 0.0020 | −7.21 | significant (worse) |
| Actor / filtered | H2GCN | −2.66 | 0.0003 | 0.0020 | 0.0039 | −1.81 | significant (worse) |


![Paired comparisons, accuracy](figures/statistical_comparisons_accuracy.png)


**Interpretation of the statistics.**

The statistical tests support three main conclusions:

1. **Original data:** all nine graph-model comparisons are significant after Holm correction. Effect sizes are very large (`|d_z| = 3.6–6.7`), and the direction of the effect is consistent across all ten splits. With ten pairs, the Wilcoxon p-value of 0.002 is the smallest attainable value.
2. **Filtered data:** most comparisons show no significant accuracy difference from the MLP. The only accuracy gain that survives Holm correction is H2GCN on filtered Squirrel (+3.15 pp, adjusted `p = 0.0058`). H2GCN on filtered Chameleon is positive but not significant after correction (`p = 0.129`).
3. **Accuracy versus macro-F1:** on Actor, H2GCN is statistically tied with the MLP in accuracy but is 2.4–2.7 F1 points lower. Its errors are concentrated in minority classes. GCN and GAT are worse on both metrics. On filtered Squirrel, GAT is significantly worse in macro-F1 despite near-identical accuracy.

The distinction between statistical and practical significance matters. The raw-data gains of about 12 pp are both statistically strong and substantial. The surviving filtered-data gain is smaller and dataset-specific. Several uncorrected p-values below 0.05 do not survive Holm correction, so they should not be presented as significant.


---


## 8. Ablations (H2GCN branches)


We compare the full H2GCN with two variants: `no_two_hop`, which removes the exact two-hop branch and keeps only `A1` diffusion, and `no_one_hop`, which removes the one-hop branch and keeps only `A2` diffusion. The two variants are trained on Chameleon and Squirrel, under both data conditions and across ten splits, giving 80 additional runs. The full-model results are reused from the primary experiment. Both ablation variants have 256-dimensional embeddings (`4 × 64`) and 1,600 fewer parameters (150,149 on Chameleon and 135,045 on Squirrel).

| Condition | Full H2GCN | no two-hop | no one-hop |
|---|---|---|---|
| Chameleon / raw | 0.6048 ± 0.0287 | 0.5976 ± 0.0298 | 0.4985 ± 0.0291 |
| Chameleon / filtered | 0.4133 ± 0.0353 | 0.4251 ± 0.0303 | 0.3971 ± 0.0352 |
| Squirrel / raw | 0.4212 ± 0.0163 | 0.4270 ± 0.0114 | 0.3636 ± 0.0188 |
| Squirrel / filtered | 0.3789 ± 0.0171 | 0.3699 ± 0.0214 | 0.3923 ± 0.0206 |


**Accuracy, mean ± SD over ten splits.**

| Condition | Variant | Mean Δ accuracy (pp) | 95 % CI (pp) | p (raw) | p (Holm) | p (Wilcoxon) | d_z | Verdict |
|---|---|---|---|---|---|---|---|---|
| Chameleon / raw | no two-hop | −0.72 | [−2.17, +0.72] | 0.287 | 0.536 | 0.266 | −0.36 | not significant |
| Chameleon / raw | no one-hop | −10.64 | [−12.64, −8.63] | < 0.0001 | < 0.0001 | 0.0020 | −3.79 | significant |
| Chameleon / filtered | no two-hop | +1.18 | [−0.75, +3.11] | 0.200 | 0.536 | 0.193 | 0.44 | not significant |
| Chameleon / filtered | no one-hop | −1.63 | [−3.27, +0.01] | 0.051 | 0.051 | 0.055 | −0.71 | not significant |
| Squirrel / raw | no two-hop | +0.58 | [−0.32, +1.47] | 0.179 | 0.536 | 0.152 | 0.46 | not significant |
| Squirrel / raw | no one-hop | −5.76 | [−7.49, −4.04] | < 0.0001 | 0.0001 | 0.0020 | −2.39 | significant |
| Squirrel / filtered | no two-hop | −0.91 | [−1.60, −0.21] | 0.016 | 0.064 | 0.027 | −0.94 | not significant |
| Squirrel / filtered | no one-hop | +1.33 | [+0.22, +2.45] | 0.024 | 0.048 | 0.029 | 0.86 | significant |


*Paired differences against the full H2GCN on matching splits; Holm correction over

the eight tests.*


Macro-F1 shows the same overall pattern. Removing the one-hop branch lowers macro-F1 by 11.97 pp on raw Chameleon (`p_Holm < 0.0001`) and 6.01 pp on raw Squirrel (`p_Holm = 0.0004`). The change is not significant on filtered Chameleon (−2.59 pp; `p_Holm = 0.118`) or filtered Squirrel (+0.68 pp). Removing the two-hop branch changes macro-F1 by between −0.88 and +1.35 pp across the four conditions; none of these differences is significant. Thus, the one-hop branch accounts for most of the measurable ablation effect on the original data, while the two-hop branch has no consistent effect.

![H2GCN ablations](figures/ablation_comparison.png)


**Which component contributes?**

- **The one-hop branch matters on the original benchmarks.** Removing it reduces accuracy by 10.6 pp on Chameleon and 5.8 pp on Squirrel. On raw Chameleon, the resulting score is close to the MLP baseline (0.4985 versus 0.5009).
- **The two-hop branch has no consistent benefit.** Removing it changes accuracy by less than 1.2 pp in every condition, and none of these differences survives Holm correction. The filtered Squirrel comparison comes closest (`p_Holm = 0.064`); on filtered Chameleon, the one-hop-only variant is slightly better, though not significantly.
- **Filtered Squirrel is an exception worth noting.** The two-hop-only variant (`no_one_hop`) scores 1.33 pp above the full model, and this difference is significant after correction (`p_Holm = 0.048`). This does not establish a general advantage for two-hop aggregation, but it suggests that its usefulness can depend on the dataset and condition.

One possible explanation is that a Squirrel node's exact two-hop neighbourhood contains hundreds of nodes. Averaging over such a large set may produce a representation close to the global class distribution, limiting its discriminative value. In this implementation and protocol, the more reliable contribution comes from retaining the node's own features and aggregating its one-hop neighbours. The ablation results are reported as observed, rather than modifying the architecture to fit the initial hypothesis.


---


## 9. Embedding analysis


For split 0, we export the representation immediately before the classifier for every model, dataset and condition (24 files in `results/embeddings/`). We project each representation to two dimensions with t-SNE (`perplexity = 30`, `init = "pca"`, seed 42) after standardising its features. PCA is used only to initialise t-SNE, not as a preprocessing step. Each panel is fitted independently, so its axes and coordinates cannot be compared directly with those in other panels.

![t-SNE, Chameleon raw](figures/tsne_chameleon_raw.png)


![t-SNE, Chameleon filtered](figures/tsne_chameleon_filtered.png)


![t-SNE, Squirrel raw](figures/tsne_squirrel_raw.png)


![t-SNE, Squirrel filtered](figures/tsne_squirrel_filtered.png)

| Condition | Model | Embedding dim | 10-NN label agreement | Silhouette |
|---|---|---|---|---|
| Chameleon / raw | MLP | 64 | 0.592 | +0.047 |
| Chameleon / raw | GCN | 64 | 0.567 | −0.049 |
| Chameleon / raw | GAT | 64 | 0.604 | −0.014 |
| Chameleon / raw | H2GCN | 576 | 0.648 | −0.024 |
| Chameleon / filtered | MLP | 64 | 0.448 | +0.030 |
| Chameleon / filtered | GCN | 64 | 0.410 | −0.006 |
| Chameleon / filtered | GAT | 64 | 0.398 | −0.005 |
| Chameleon / filtered | H2GCN | 576 | 0.429 | −0.002 |
| Squirrel / raw | MLP | 64 | 0.400 | −0.002 |
| Squirrel / raw | GCN | 64 | 0.394 | −0.064 |
| Squirrel / raw | GAT | 64 | 0.428 | −0.033 |
| Squirrel / raw | H2GCN | 576 | 0.451 | −0.074 |
| Squirrel / filtered | MLP | 64 | 0.383 | +0.002 |
| Squirrel / filtered | GCN | 64 | 0.341 | −0.049 |
| Squirrel / filtered | GAT | 64 | 0.301 | −0.038 |
| Squirrel / filtered | H2GCN | 576 | 0.374 | −0.021 |
| Actor / raw | MLP | 64 | 0.354 | −0.013 |
| Actor / raw | GCN | 64 | 0.252 | −0.050 |
| Actor / raw | GAT | 64 | 0.238 | −0.032 |
| Actor / raw | H2GCN | 576 | 0.304 | −0.030 |
| Actor / filtered | MLP | 64 | 0.353 | −0.012 |
| Actor / filtered | GCN | 64 | 0.255 | −0.070 |
| Actor / filtered | GAT | 64 | 0.240 | −0.033 |
| Actor / filtered | H2GCN | 576 | 0.306 | −0.034 |


*Split-0 representations. The 10-NN label agreement is the fraction of the ten

nearest neighbours (standardised embedding space, Euclidean) sharing the node label;

the silhouette is computed on at most 2 000 nodes. Both are computed by

`viz.tsne_plotter.embedding_quality` and stored in

`results/analysis/embeddings_index.json`.*


**Reading.***

Silhouette scores are close to zero across all models and conditions (−0.074 to +0.047), so the embeddings do not form clearly separated global class clusters. This is consistent with the strong heterophily and modest absolute classification scores.

The 10-nearest-neighbour label agreement broadly follows test performance in several conditions. For example, on raw Chameleon, GAT (0.604) and H2GCN (0.648) exceed the MLP (0.592); on Actor, the MLP (0.354) is well above GCN (0.252) and GAT (0.238). However, H2GCN has the highest local agreement on raw Chameleon and Squirrel without always achieving the highest accuracy. Local neighbour purity alone therefore does not explain classification performance.

The t-SNE plots are qualitative diagnostics only. They do not prove oversmoothing or establish that one model is superior, and distances between clusters in independently fitted panels are not meaningful.


---


## 10. Discussion

### 1. Does graph structure improve performance over the MLP?

It does on the original Chameleon and Squirrel benchmarks: every graph model gains 8.8–13.1 accuracy points, and all nine comparisons remain significant after Holm correction. On the filtered versions, most differences disappear; GCN is slightly worse on both datasets, and four of the six graph-model comparisons are statistically indistinguishable from the MLP. On Actor, GCN and GAT are 5.4–6.1 points worse. The evidence therefore supports a conditional conclusion: graph structure helps on the original Chameleon/Squirrel splits, but its contribution on the filtered datasets is small and dataset-dependent. This pattern is consistent with the concerns raised by Platonov et al. (2023), but our design cannot attribute the entire change to leakage because filtering also changes the nodes and splits.

### 2. When does GCN help or hurt?

GCN has the largest change in behaviour across conditions. It gains 12–13 pp on raw Chameleon and Squirrel, is effectively tied with the MLP on their filtered versions, and loses 5.4–5.7 pp on Actor. These results suggest that aggregation helps when neighbourhoods provide useful information beyond node features. When the graph carries little label signal, aggregation can instead dilute the node's own representation. Actor is the clearest example: it has an average degree of 7 and 26% of nodes belong to feature-identical groups.

### 3. Does GAT improve on GCN?

No consistent advantage appears. GAT is 0.7 pp above GCN on raw Chameleon, 3.1 pp below it on raw Squirrel, slightly worse on Actor, and within 1 pp of it on filtered data. Attention changes how neighbours are weighted, but it cannot recover label information that the neighbourhood does not contain.

### 4. Does H2GCN benefit from higher-order neighbourhoods?

H2GCN is the most robust graph model on the filtered datasets: it gains 3.76 pp on filtered Chameleon (not significant after correction), 3.15 pp on filtered Squirrel (significant), and is tied with the MLP in Actor accuracy. The ablations suggest that this robustness does not generally come from the exact two-hop branch. Removing that branch changes accuracy by at most 1.2 pp and is never significant after correction. Removing the one-hop branch, by contrast, costs 10.6 pp on raw Chameleon and 5.8 pp on raw Squirrel. Filtered Squirrel is the exception: the two-hop-only variant performs 1.33 pp above the full model (`p_Holm = 0.048`).

H2GCN's Actor accuracy also needs qualification. Its macro-F1 is 2.4–2.7 points below the MLP because it predicts the smallest class for only 0.2% of test nodes (F1 = 0.014), compared with the MLP's F1 of 0.222. On the largest class, however, H2GCN is slightly better (0.585 versus 0.573). Its deficit is therefore concentrated in minority-class prediction rather than spread evenly across classes.

### 5. What changes after filtering?

Most apparent gains from message passing disappear after filtering. On Chameleon, accuracy falls by 19–27 pp for the graph models, compared with 12.5 pp for the MLP. On Squirrel, graph-model accuracy falls by 4–12 pp, while MLP accuracy increases by 1.5 pp. The model ranking changes as well: GAT leads on raw Chameleon, whereas H2GCN leads on filtered Chameleon and Squirrel. This agrees with Platonov et al.'s observation that results on the original benchmarks may not transfer to the filtered versions.

These comparisons do not isolate the effect of duplicate removal. Filtering changes the node population, class balance (for Squirrel, max/min rises from 1.00 to 3.24) and degree distribution. The observed differences may therefore have several causes.

### 6. How much train–test feature overlap remains, and what does it mean?

The share of test nodes with an exact feature twin in the training set is 14.45% on raw Chameleon, 4.21% on raw Squirrel and 22.34% on Actor. It remains 10.81% on filtered Chameleon and 5.83% on filtered Squirrel. These values should be interpreted cautiously: our audit uses feature similarity, not the target-and-neighbourhood criterion from Platonov et al.; only 6.2%–19.7% of duplicate nodes belong to label-consistent groups; and we do not measure whether a model's correct predictions were caused by feature overlap. The audit identifies a potential weakness in the benchmarks, not a direct measurement of leakage exploited by the models.

### 7. Are the results consistent across datasets?

No. Graph models gain about 12 pp on raw Chameleon and Squirrel, lose 5–6 pp on Actor in the case of GCN and GAT, and are mostly tied with the MLP on filtered Chameleon and Squirrel. A broad claim that GNNs always help or always hurt under heterophily is not supported by these experiments.

### 8. Are the differences statistically significant?

All nine comparisons on the original data survive Holm correction. On filtered data, the only significant accuracy gain is H2GCN on Squirrel. GAT is significantly worse in macro-F1 on filtered Squirrel, and all three graph models are significantly worse than the MLP in macro-F1 on Actor.

### 9. Are the differences practically meaningful?

The gains of around 12 pp on the original data are substantial. The filtered-data gain that survives correction is smaller (3.15 pp) and specific to Squirrel. The 5–6 pp accuracy losses and 10–12 pp macro-F1 losses on Actor are practically important, not merely statistical fluctuations.

### 10. What remains unexpected?

Three findings stand out. First, MLP accuracy drops from 0.501 on raw Chameleon to 0.376 on the filtered version even though the feature dimension is unchanged; removing nodes also removes some relatively easy test examples. Second, accuracy can hide severe minority-class failures: on filtered Squirrel, GAT assigns 74.5% of predictions to the majority class and has F1 scores of only 0.074–0.122 on the other classes. On raw Actor, GCN and GAT never predict the smallest class. Third, H2GCN shows a milder version of the Actor minority-class problem, but has the strongest minority-class F1 scores on filtered Squirrel.

## 11. Limitations

- **Dataset coverage.** The study uses three small-to-medium benchmarks from two domains (Wikipedia pages and film co-occurrence). They do not represent all forms of heterophily. Their adjusted homophily is close to zero, so the results should not be generalised to graphs with moderate heterophily or informative neighbourhoods.
- **Split comparability.** Raw and filtered Chameleon/Squirrel have different node populations. Actor uses the same graph in both conditions but different masks. Comparisons between conditions are descriptive; statistical tests are restricted to models evaluated on the same condition and split.
- **Preprocessing.** `actor/filtered` is a custom reproduction with stratified splits, not a duplicate-filtered dataset. No upstream filtered Actor version is available. All `raw` conditions use the pinned Geom-GCN files.
- **H2GCN implementation.** Our implementation follows the model's main principles but differs from the original code: it uses row-normalised diffusion, one dropout on the concatenated representation and a linear classifier. It is not a bit-exact reproduction of Zhu et al. (2020), so published results may not be directly comparable.
- **Statistical power.** Ten splits provide limited power. The smallest attainable Wilcoxon p-value is 0.002, and Holm correction across 18 tests is conservative. Effect sizes apply to this protocol; raw p-values below 0.05 that do not survive correction are treated as non-significant.
- **Reproducibility.** Seeds are fixed per split and shared across models. Dense CPU operations are deterministic, but PyTorch sparse scatter reductions are not guaranteed to be bitwise identical across builds; reruns may differ in the final decimal places.
- **Compute budget.** Experiments ran on a CPU with four physical cores and took about three hours. More compute would allow additional splits, seeds and independent initialisations per split.
- **Metric coverage.** We report accuracy and macro-F1, but not calibration, robustness or fairness metrics. The Actor results show why using accuracy alone would be insufficient.
- **Embedding diagnostics.** t-SNE is stochastic and fitted independently for each model, so panel coordinates are not comparable. Visual separation does not establish oversmoothing. The kNN and silhouette scores provide quantitative context, but they are also dependent on the standardised embedding space. We use t-SNE rather than UMAP because `umap-learn` is not part of the pinned environment.
- **PDF export.** The pinned environment does not include LaTeX or Pandoc, so the report is delivered in Markdown rather than PDF (see section 14).

## 12. Conclusion

Across 240 primary runs and 80 ablations, the apparent benefit of graph neural networks depends strongly on the benchmark condition. On the original Chameleon and Squirrel splits, every graph model beats the feature-only MLP by 8.8–13.1 accuracy points. On the filtered versions, most of this advantage disappears: GCN and GAT are generally tied with the MLP, while H2GCN retains a significant gain only on filtered Squirrel. On Actor, GCN and GAT lose 5–6 accuracy points and 10–12 macro-F1 points; H2GCN matches the MLP in accuracy but remains weaker on minority classes.

The H2GCN ablations point to ego-preservation and one-hop aggregation as the main sources of its performance on the original benchmarks. The exact two-hop branch provides no consistent benefit, with filtered Squirrel as the only condition showing a significant result in favour of the two-hop-only variant.

The main practical lessons are:

1. **Always include a feature-only MLP baseline** to measure what graph structure actually contributes.
2. **Audit benchmark duplicates and feature overlap** before drawing conclusions about GNN performance.
3. **Evaluate accuracy and macro-F1 together** and inspect per-class predictions, especially on imbalanced datasets.
4. **Avoid broad claims based on one benchmark.** The effect of message passing varies substantially across datasets and data conditions.

The results are consistent with concerns about benchmark leakage, but they do not prove that leakage alone explains the raw-versus-filtered performance gap.

## 13. References


1. Zhu, J., Yan, Y., Zhao, L., Heimann, M., Akoglu, L., Koutra, D. (2020).

   *Beyond Homophily in Graph Neural Networks: Current Limitations and Effective

   Designs.* NeurIPS 2020. arXiv:2006.11468.

2. Platonov, O., Kuznedelev, D., Diskin, M., Babenko, A., Prokhorenkova, L. (2023).

   *A Critical Look at the Evaluation of GNNs under Heterophily: Are We Really

   Making Progress?* ICLR 2023. arXiv:2302.11640. Data:

   `https://github.com/yandex-research/heterophilous-graphs` (revision

   `a431395582e929d88271309716bea4fe24ce6318`).

3. Pei, H., Wei, B., Chang, K. C.-C., Lei, Y., Yang, B. (2020). *Geom-GCN: Geometric

   Graph Convolutional Networks.* ICLR 2020. Data: Geom-GCN repository (revision

   `f1fc0d14b3b019c562737240d06ec83b07d16a8f`).

4. Kipf, T. N., Welling, M. (2017). *Semi-Supervised Classification with Graph

   Convolutional Networks.* ICLR 2017.

5. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., Bengio, Y.

   (2018). **Graph Attention Networks.** ICLR 2018.

6. Lim, D., Hohne, F., Li, X., Huang, S. L., Gupta, V., Bhalerao, O., Lim, S.-N.

   (2021). *Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and

   Strong Simple Methods.* NeurIPS 2021.

7. Rozemberczki, B., Allen, C., Sarkar, R. (2021). *Multi-scale Attributed Node

   Embedding.* Journal of Complex Networks 9(2).

8. Fey, M., Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch

   Geometric.* ICLR Workshop. Documentation:

   `https://pytorch-geometric.readthedocs.io`.

9. Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep

   Learning Library.* NeurIPS 2019.

10. Pedregosa, F. et al. (2011). **Scikit-learn: Machine Learning in Python.** JMLR

    12:2825–2830.

11. Virtanen, P. et al. (2020). *SciPy 1.0: Fundamental Algorithms for Scientific

    Computing in Python.* Nature Methods 17:261–272.

12. van der Maaten, L., Hinton, G. (2008). **Visualizing Data using t-SNE.** JMLR

    9:2579–2605.

