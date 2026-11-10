# A2 — Heterophily: implementation audit and development plan

Branch: `feature/a2-complete-experiment`
Baseline commit: `2f53238` (`main`, unchanged by this work)

## 1. Scope of the assignment

Compare a structure-blind baseline (MLP) with three graph models (GCN, GAT, H2GCN) on
Chameleon, Squirrel and Actor under two data conditions (`raw`, `filtered`), ten
splits each: `4 x 3 x 2 x 10 = 240` primary training runs, plus H2GCN ablations,
a duplicate/leakage audit, statistical analysis, figures and a research report.

## 2. What already existed (inventory)

| Path | Status | Notes |
|---|---|---|
| `requirements.txt` | PARTIAL | Pinned Python 3.12 CPU stack (`torch==2.8.0+cpu`, `torch-geometric==2.7.0`). Missing the scientific stack that the assignment requires (`scipy`, `matplotlib`, `scikit-learn`) and the test runner (`pytest`). |
| `.gitignore` | COMPLETE | Already ignores `.venv/`, `.cache/`, `__pycache__/`, `datasets/`. Extended only with generated run artifacts. |
| `data/loader.py` | PARTIAL | `get_dataset(name, mode, root, offline, rebuild)` works for `chameleon/{raw,filtered}`, `squirrel/{raw,filtered}`, `actor/raw` (verified from `reports/data_manifest.json` and the pinned Geom-GCN / Platonov revisions). `actor/filtered` is absent, so the required 3 datasets x 2 modes grid cannot be built. |
| `data/validation.py` | COMPLETE | Validates graph shape, self-loops, duplicate edges, undirectedness, `[N, 10]` boolean masks and disjoint partitions. Reused unchanged. |
| `analysis/homophily.py` | COMPLETE | Edge / node / degree-weighted adjusted homophily with documented conventions and `None` for undefined values. Reused unchanged; an extra node-proportion adjusted homophily variant is added for the assignment formula (see section 5). |
| `analysis/duplicate_audit.py` | MISSING | Required duplicate and train-test leakage audit does not exist. |
| `models/` | MISSING | No model implementations. |
| `engine/` | MISSING | No early stopping or training loop. |
| `viz/` | MISSING | No plotting / t-SNE modules. |
| `scripts/prepare_data.py` | PARTIAL | Works, but hard-codes "five conditions" and "Actor filtered is not configured". Must be updated to six conditions. |
| `scripts/analyze_graphs.py` | COMPLETE | Descriptive statistics + homophily summary; reused unchanged apart from the condition list coming from the loader. |
| `reports/data_manifest.json`, `reports/data_inventory.csv`, `reports/dataset_summary.csv` | COMPLETE | Produced by the previous milestone; retained and regenerated. |
| `README.md` | PARTIAL | Documents dataset preparation only. |
| `tests/` | MISSING | No test suite in the repository. |

No user modifications were present in the working tree at branch creation
(`git status` clean), so nothing had to be preserved beyond the committed files.

## 3. Verified dataset facts (from the committed manifest and the sources)

| Condition | Nodes | Features | Classes | Undirected edges | Avg degree | Split source |
|---|---|---|---|---|---|---|
| chameleon/raw | 2277 | 2325 | 5 | 31371 | 27.55 | Geom-GCN `chameleon_split_0.6_0.2_{0..9}` |
| chameleon/filtered | 890 | 2325 | 5 | 8854 | 19.90 | Platonov et al. `chameleon_filtered.npz` masks |
| squirrel/raw | 5201 | 2089 | 5 | 198353 | 76.27 | Geom-GCN `squirrel_split_0.6_0.2_{0..9}` |
| squirrel/filtered | 2223 | 2089 | 5 | 46998 | 42.28 | Platonov et al. `squirrel_filtered.npz` masks |
| actor/raw | 7600 | 932 | 5 | 26659 | 7.02 | Geom-GCN `film_split_0.6_0.2_{0..9}` |
| actor/filtered | 7600 | 932 | 5 | 26659 | 7.02 | Platonov et al. `actor.npz` alternative masks (added below) |

* Platonov et al. removed duplicate nodes only for Squirrel and Chameleon
  ("nodes with no incoming edges and for which there exists a node in the graph with
  the same average monthly traffic and the same set of outgoing edges"), and provide
  new 10 x (50/25/25) masks for the filtered graphs. Their `data/` folder at revision
  `a431395582e929d88271309716bea4fe24ce6318` contains `chameleon_filtered.npz`,
  `squirrel_filtered.npz` and `actor.npz`, but **no** `actor_filtered.npz`.
* `actor.npz` has 7600 nodes / 26659 undirected edges, which equals the Geom-GCN
  Actor graph after self-loop removal and symmetrisation (53318 directed edges).
  Inspecting the masks showed that `actor.npz` reproduces the original Geom-GCN
  masks, and there is no duplicate-removed Actor file upstream.
* Raw 10-split masks are the Geom-GCN `0.6/0.2/0.2` masks, not freshly generated
  random splits. Raw and filtered node populations are different for
  Chameleon/Squirrel, so raw and filtered results are **not** matched pairs.

## 4. Decisions taken

1. **`actor/filtered`**: added to `SUPPORTED_DATASETS`. Because no upstream
   duplicate-removed Actor dataset exists and the distributed `actor.npz` reuses the
   original Geom-GCN masks, this condition keeps the unchanged Platonov Actor graph
   and uses ten **custom class-stratified 50/25/25 splits** generated from the
   documented seed `20230222`. It is labelled a *custom reproduction* in the dataset
   metadata (`split_source`, `split_seed`, `duplicate_removal`) and the duplicate
   audit reports the remaining Actor duplicates (26% of nodes) independently.
2. **Dependencies**: `scipy` (required paired tests), `matplotlib` (figures),
   `scikit-learn` (`f1_score`, `TSNE`), `pytest` (tests) are added and pinned. No
   existing version is upgraded.
3. **Configuration format**: JSON (`configs/full.json`) instead of YAML, because the
   project has no YAML parser and the assignment forbids unnecessary dependency
   additions. `scripts/run_all.py --config configs/full.json` is the single entry
   point.
4. **`main` is untouched**; every milestone is committed to
   `feature/a2-complete-experiment` and the merge is left to the user.

## 5. Compatibility decisions

* Existing public interfaces (`get_dataset`, `validate_dataset`, `homophily_metrics`,
  `summarize_dataset`, `scripts.prepare_data.main`, `scripts.analyze_graphs.main`)
  are preserved. `actor/filtered` is added additively.
* `adjusted_homophily` keeps its original degree-weighted definition (Lim et al.
  convention, as documented in the existing docstring). The assignment's
  node-proportion formula is added as a separate key
  `adjusted_homophily_node_proportions` so both conventions are available.
* H2GCN embedding width is `9 x hidden` for the full model and is intentionally not
  forced to match the hidden size of the other models.

## 6. Planned milestones

| # | Milestone | Artifacts |
|---|---|---|
| 1 | Audit and plan | this document |
| 2 | Models | `models/mlp.py`, `models/vanilla_gnn.py`, `models/h2gcn.py`, `models/registry.py` |
| 3 | Data | `actor/filtered` in `data/loader.py`, updated `scripts/prepare_data.py` |
| 4 | Analysis | `analysis/duplicate_audit.py`, `analysis/stats.py`, homophily addition |
| 5 | Engine | `engine/early_stopping.py`, `engine/trainer.py` |
| 6 | Experiments | `scripts/run_experiments.py`, `scripts/run_ablations.py`, `configs/full.json` |
| 7 | Tests | `tests/` |
| 8 | Results | `results/results.{json,csv}`, `results/ablation_results.{json,csv}`, embeddings, logs |
| 9 | Analysis + figures | `results/analysis/*.csv`, `reports/figures/*.png` |
| 10 | Report | `reports/FINAL_REPORT.md` (+PDF if a converter exists) |
| 11 | Reproducibility | README, `scripts/run_all.py` |
