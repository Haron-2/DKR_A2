# A2 вЂ” Heterophily: When Neighbors Mislead

Reproducible comparison of a structure-blind baseline (MLP) with three graph neural
networks (GCN, GAT, H2GCN) on Chameleon, Squirrel and Actor, in the original
(`raw`) and duplicate-audited / corrected-split (`filtered`) conditions.

* Branch: `feature/a2-complete-experiment` (the merge into `main` is done by the user)
* Report: [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md)
* Results: [`results/results.json`](results/results.json), [`results/results.csv`](results/results.csv)
* Figures: [`reports/figures/`](reports/figures)

## 1. Research question

Do graph neural networks actually exploit structure on heterophilous graphs, or do
aggregation-based models merely re-use node features while being degraded by
neighbourhood mixing? The experiment measures the *Structural Delta*
`Accuracy(GNN) в€’ Accuracy(MLP)` per split, checks whether it is statistically
significant after multiple-testing correction, and quantifies how much of the
original benchmark advantage depends on duplicate nodes and train-test leakage.

## 2. Project description

* 4 models Г— 3 datasets Г— 2 data conditions Г— 10 splits = **240 primary runs**.
* 2 H2GCN branch ablations Г— 2 datasets Г— 2 conditions Г— 10 splits = **80 ablation runs**.
* Every run trains a fresh model on one split, early-stops on validation loss,
  restores the best checkpoint and evaluates the test mask exactly once.
* A duplicate / leakage audit, paired statistical tests, figures and a written
  report are produced from the stored artifacts.

## 3. Architecture overview

| Model | File | Summary |
|---|---|---|
| MLP | `models/mlp.py` | Linear в†’ LayerNorm в†’ ReLU в†’ Dropout(0.5) в†’ Linear. Ignores `edge_index`. |
| GCN | `models/vanilla_gnn.py` | Two `GCNConv` layers, self-loops and symmetric normalisation, ReLU + Dropout(0.5). |
| GAT | `models/vanilla_gnn.py` | `GATConv(8 heads Г— 8 channels)` в†’ ELU в†’ Dropout в†’ single-head output. |
| H2GCN | `models/h2gcn.py` | Ego/neighbour separation, exact one-hop and two-hop diffusion, two rounds, concatenation `[H0, R1(1), R2(1), R1(2), R2(2)]`, linear classifier. |

All models expose `forward(x, edge_index) -> logits` and
`get_embeddings(x, edge_index) -> embeddings` (the representation fed to the
classifier). Embedding widths: MLP/GCN/GAT = 64, H2GCN = 576 (9 Г— 64).

## 4. Datasets

| Condition | Nodes | Features | Classes | Undirected edges | Splits |
|---|---|---|---|---|---|
| chameleon/raw | 2277 | 2325 | 5 | 31371 | Geom-GCN `0.6/0.2/0.2`, 10 splits |
| chameleon/filtered | 890 | 2325 | 5 | 8854 | Platonov et al. corrected masks |
| squirrel/raw | 5201 | 2089 | 5 | 198353 | Geom-GCN `0.6/0.2/0.2`, 10 splits |
| squirrel/filtered | 2223 | 2089 | 5 | 46998 | Platonov et al. corrected masks |
| actor/raw | 7600 | 932 | 5 | 26659 | Geom-GCN `film` splits |
| actor/filtered | 7600 | 932 | 5 | 26659 | custom stratified 50/25/25, seed 20230222 |

* `raw` = original Geom-GCN benchmark graphs and their ten splits.
* `filtered` = Platonov et al. (2023) duplicate-audited redistributions for
  Chameleon and Squirrel. For Actor no duplicate-removed dataset exists upstream
  (their `actor.npz` reproduces the original masks), so `actor/filtered` is a
  **custom reproduction**: unchanged graph, ten class-stratified splits from a
  documented seed. It is labelled as such in the dataset metadata.
* Raw and filtered splits are **not paired** for Chameleon/Squirrel (different
  node populations); raw-vs-filtered comparisons are descriptive only.

## 5. Repository structure

```text
analysis/      homophily metrics, duplicate/leakage audit, paired statistics
configs/       full.json (datasets, seeds, hyperparameters, ablation grid)
data/          dataset loading, six conditions, split validation
docs/          implementation audit and development plan
engine/        early stopping, training loop, run orchestration helpers
models/        MLP, GCN, GAT, H2GCN and the model registry
reports/       final report, figures, tables, dataset statistics
results/       results.json/csv, ablation results, analysis tables, logs
scripts/       data preparation, experiments, ablations, analysis, figures
tests/         model, graph-operation, dataset, metric, training, integration tests
viz/           figures and t-SNE embedding plots
```


## 6. Installation

Python 3.12 is required (the pinned CPU wheels target 3.12).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

All commands below assume the repository root as the working directory and the
virtual-environment interpreter. Under a POSIX shell use `.venv/bin/python`.

## 7. Pinned dependencies

`requirements.txt` pins the complete CPU stack: `torch==2.8.0+cpu`,
`torch-geometric==2.7.0`, `numpy==2.3.4`, `scipy==1.18.1`,
`scikit-learn==1.9.1`, `matplotlib==3.11.2`, `pytest==9.2.0` (plus transitive
dependencies). No existing version was upgraded; `scipy`, `matplotlib`,
`scikit-learn` and `pytest` were added for the paired tests, figures, metrics and
test suite required by the assignment.

## 8. Dataset preparation

```powershell
.venv\Scripts\python.exe -m scripts.prepare_data
```

This downloads the pinned Geom-GCN and Platonov et al. files into `datasets/`
(git-ignored), validates every graph and all ten mask columns, and writes
`reports/data_manifest.json`, `reports/data_inventory.csv` and
`reports/dataset_summary.csv`. Add `--offline` to reuse existing raw files and
`--rebuild` to reprocess them.

## 9. Tests

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

64 tests cover the model interfaces, H2GCN adjacency construction, dataset and
mask validation, homophily metrics on hand-computed graphs, duplicate / feature
overlap detection (including the dataset label written by the audit), training-mask
isolation, early-stopping restoration, result serialization, resumability,
embeddings and figure generation.

## 10. Smoke experiment

```powershell
.venv\Scripts\python.exe -m scripts.run_all --smoke --skip-data
```

Runs the whole pipeline on `chameleon/raw`, split 0, 15 epochs, into
`results/smoke/` and `reports/figures_smoke/` (both git-ignored).

## 11. Full experiment (the one documented main command)

```powershell
.venv\Scripts\python.exe -m scripts.run_all --config configs/full.json
```

This orchestrates, in order: dataset preparation, graph statistics, the duplicate
and feature-overlap audit, the 240 primary runs, the 80 ablation runs, the
statistical analysis, the per-class diagnostics (inference only, from the stored
checkpoints), the Markdown tables and all figures, followed by an artifact /
completeness check. Every step is a separate process; re-running the command resumes
skipping completed runs.

Individual steps:

```powershell
.venv\Scripts\python.exe -m scripts.run_experiments --config configs/full.json --threads 8
.venv\Scripts\python.exe -m scripts.run_ablations   --config configs/full.json --threads 8
```

`scripts.run_experiments` accepts `--datasets`, `--modes`, `--models`, `--splits`,
`--max-epochs`, `--limit`, `--force`, `--no-embeddings`, `--output-dir` and
`--print-parameters`. Results are written after every run, so an interrupted
session resumes from the last completed run.

## 12. Ablation experiments

The ablation grid is part of `configs/full.json` (`ablations` section) and runs
`H2GCN[no_two_hop]` and `H2GCN[no_one_hop]` on Chameleon and Squirrel, both
conditions, ten splits each, with identical seeds and hyperparameters. The full
H2GCN is not re-run; its primary results are paired with the variants during
analysis.

## 13. Analysis and visualisation

```powershell
.venv\Scripts\python.exe -m scripts.audit_duplicates --offline
.venv\Scripts\python.exe -m scripts.analyze_results
.venv\Scripts\python.exe -m scripts.evaluate_predictions
.venv\Scripts\python.exe -m scripts.export_tables
.venv\Scripts\python.exe -m scripts.make_figures
```

* `audit_duplicates` writes `results/analysis/duplicate_audit.json` (and `.csv`
  with one row per split). It reports train-test **feature overlap**, not a direct
  measurement of label transfer; the JSON states this explicitly.
* `analyze_results` writes `results/analysis/summary.json` plus per-metric CSV
  tables (aggregates, Structural Delta, paired tests, raw-vs-filtered).
* `evaluate_predictions` reloads every stored best checkpoint in inference mode (no
  retraining) and writes `per_class_metrics.csv`, `per_class_summary.csv` and
  `confusion_matrices.json`; the recomputed accuracies reproduce the recorded ones.
* `export_tables` renders the Markdown tables in `reports/tables/` (including
  per-class tables for Actor raw and Squirrel filtered).
* `make_figures` writes the PNG figures in `reports/figures/` (and the t-SNE
  coordinates as CSV next to them).


## 14. Report generation

The report is committed as a static Markdown document whose numbers come from the
generated tables:

```powershell
.venv\Scripts\python.exe -m scripts.export_tables
```

`reports/FINAL_REPORT.md` embeds the tables from `reports/tables/` and the figures
from `reports/figures/`. PDF conversion requires Pandoc
(`pandoc reports/FINAL_REPORT.md -o reports/FINAL_REPORT.pdf`), which is not part
of the pinned Python environment; see the report's limitations section.

## 15. Output descriptions

| Path | Content | Committed |
|---|---|---|
| `results/results.json` / `.csv` | 240 primary run records | yes |
| `results/ablation_results.json` / `.csv` | 80 ablation run records | yes |
| `results/run_status.json`, `results/ablation_status.json` | completeness reports | yes |
| `results/analysis/summary.json` | all aggregated tables and tests | yes |
| `results/analysis/*.csv` | aggregates, deltas, paired tests, raw-vs-filtered | yes |
| `results/analysis/duplicate_audit.json` / `.csv` | duplicate groups and train-test feature overlap | yes |
| `results/analysis/per_class_metrics.csv` | per-run, per-class precision/recall/F1 | yes |
| `results/analysis/per_class_summary.csv` | per-class metrics averaged over ten splits | yes |
| `results/analysis/confusion_matrices.json` | test-mask confusion matrix per run | yes |
| `results/analysis/embeddings_index.json` | exported embedding inventory | yes |
| `results/logs/**` | per-epoch train/validation loss per run | no (regenerable) |
| `results/checkpoints/**` | best validation checkpoint per run | no (regenerable, large) |
| `results/embeddings/**` | split-0 embeddings for all 24 model/condition pairs | no (regenerable, large) |
| `reports/figures/*.png` | final figures | yes |
| `reports/tables/*.md` | Markdown tables used by the report | yes |
| `reports/FINAL_REPORT.md` | scientific report | yes |

Regenerate the ignored artifacts with the experiment commands above; nothing in the
report depends on files that are not reproducible from `configs/full.json`.

## 16. Reproducibility details

* **Seeds.** `seed_base = 42`; the seed of split `s` is `42 + s` and is shared by
  all models compared on that split. Model initialisation, dropout and optimiser
  state are seeded, and PyTorch is restricted to `--threads` intra-op threads.
* **Splits.** Original Geom-GCN masks for `raw`; Platonov et al. masks for
  Chameleon/Squirrel `filtered`; custom stratified masks (seed `20230222`) for
  `actor/filtered`.
* **Training.** Adam (`lr=0.01`, `weight_decay=5e-4`), cross-entropy, dropout 0.5,
  early stopping on validation loss (`patience=50`, `min_delta=1e-4`,
  `max_epochs=1000`), best checkpoint restored before the single test evaluation.
* **Transductive setting.** The full graph and all features are visible; only
  training-mask labels contribute to the gradient.
* **Hardware/software.** CPU only, 4 physical cores / 15.3 GB RAM, Python 3.12,
  see `requirements.txt` for exact versions. Full run time is reported in
  `reports/FINAL_REPORT.md`.
* **Determinism.** Seeds are fixed and the pipeline is deterministic on CPU for the
  dense paths; sparse scatter operations may not be bitwise reproducible across
  PyTorch builds, so results are reported with mean В± SD over ten splits.

## 17. Known limitations

* `actor/filtered` is a custom reproduction (alternative splits, unchanged graph);
  it is not a duplicate-removed dataset.
* Raw and filtered results are not paired for Chameleon and Squirrel.
* H2GCN is a core-principles implementation, not a bit-exact reproduction of the
  authors' code; no trainable transformation is used inside the diffusion rounds.
* Ten splits per condition limit statistical power; Holm-corrected paired t-tests
  are reported together with the Wilcoxon test and effect sizes.
* t-SNE panels are qualitative and computed per model, so axes are not comparable
  across panels.
* Checkpoints, per-epoch logs and raw embeddings are kept locally only.
