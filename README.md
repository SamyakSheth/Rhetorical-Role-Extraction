# It Depends on the Task: Rethinking Rhetorical Segmentation of Dutch Judicial Decisions

Code and dataset accompanying the paper *"It Depends on the Task: Rethinking Rhetorical
Segmentation of Dutch Judicial Decisions"* (Sheth, Spanakis, van Dijck), submitted to JURIX.

This repository contains the sentence-level annotated dataset, all model-training/evaluation
notebooks, and the raw prediction outputs behind every number reported in the paper.

## Repository structure

```
data/               Clean sentence-level dataset (train/eval/test splits)
preprocessing/       Sentence-splitting script used to build the dataset from raw XML
training/            Notebooks that train each model family from scratch
notebooks/           One notebook per formulation: loads trained checkpoints, runs inference,
                     evaluates, and applies header-prior fusion — self-contained and independently runnable
predictions/         Saved model outputs (probabilities + predicted labels) for the test set
models/registry.json Model checkpoint registry 
```

## Notebook → paper mapping

| Notebook | Formulation | Produces |
|---|---|---|
| `notebooks/01_five_way_inference.ipynb` | Direct five-way classification | Table 2 & 3 "5-way" rows; error decomposition|
| `notebooks/02_pipeline_a.ipynb` | Pipeline A (Stage 1 x Stage 2) | Table 2 & 3 "Pipeline A" rows; error decomposition |
| `notebooks/03_pipeline_b.ipynb` | Pipeline B (Stage 1 x OvR) | Table 2 & 3 "Pipeline B" rows; error decomposition |
| `notebooks/04_stage1_relevance.ipynb` | None Detection (Stage 1) | Table 2 & 3 "Stage 1" rows |
| `notebooks/05_stage2_fourway.ipynb` | Four-Way Role Classification (Stage 2) | Table 2 & 3 "Stage 2 4-way" rows |
| `notebooks/06_ovr_detectors.ipynb` | One-vs-Rest Role Detection | Table 2 & 3 "OvR" rows |
| `notebooks/07_bertje_evaluation.ipynb` | All formulations, BERTje | Table 2 "BERTje" rows |

Each notebook in `notebooks/` is self-contained: it loads the RobBERT/BERTje checkpoint(s) for its
own formulation, runs inference on `data/test.csv`, evaluates against the paper's reported
metrics, sweeps the header-prior fusion weight $\lambda$ on `data/eval.csv`, and applies the tuned
value to the test set.

### Training notebooks

| Notebook | Trains |
|---|---|
| `training/robbert_train.ipynb` | All RobBERT models: five-way, Stage 1, Stage 2, and the four OvR detectors |
| `training/bertje_train.ipynb` | BERTje's five-way, Stage 1, and Stage 2 models |
| `training/logistic_regression.ipynb` | The Logistic Regression + TF-IDF lexical baseline, across the same formulations (5-way, Stage 1, Stage 2, OvR) |

Note: the training notebooks were developed and run locally with relative paths matching the
original project layout; you may need to adjust data/output paths at the top of each notebook
before re-running training end-to-end in this repository's directory structure. The `notebooks/`
inference notebooks are already verified to run correctly against `data/` and `predictions/` as
laid out here.

## Reproducing the results

```bash
conda env create -f environment.yml
conda activate rhetorical-roles-nl
jupyter notebook notebooks/
```

Each notebook in `notebooks/` can be run independently and top-to-bottom against the checkpoints
referenced in `models/registry.json`. `predictions/` already contains the saved outputs used for
the paper's reported numbers. To train models from scratch instead of using existing checkpoints, see
`training/` (and the path-adjustment note above).

## Model weights

Model checkpoints are hosted publicly on Huggingface. `models/registry.json` documents the model keys,
base checkpoint (`pdelobelle/robbert-v2-dutch-base`), and task each one performs; the `hf_repo`
field contains the link to the folders on huggingface. `models/'best_models_registry.json` is the registry used locally during experiments it contains the metadata for each of the model configurations

## Dataset

See [`data/README.md`](data/README.md) for the full data card (schema, splits, label
distribution). Raw XML source files and annotation-tool exports are not included — only the
final, cleaned sentence-level dataset used in all experiments.

## Citation

```
TODO: add citation once published
```

## License

TODO — see [`LICENSE`](LICENSE). Not yet finalized.
