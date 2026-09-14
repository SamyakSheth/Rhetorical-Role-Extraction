# Dataset

Sentence-level rhetorical-role annotations for 100 Dutch judicial decisions sampled from
[Rechtspraak.nl](https://www.rechtspraak.nl/), the official public portal of the Dutch judiciary.
See the paper (Section 3) for full corpus scope, annotation methodology, and inter-annotator
agreement figures.

## Files

| File | Sentences | Role |
|---|---|---|
| `train.csv` | 5,608 | Model training |
| `eval.csv` | 1,281 | Validation set (λ tuning, early stopping) |
| `test.csv` | 1,502 | Held-out test set — all reported results |

Splits are at the case (document) level, so no sentences from the same decision appear in
more than one split.

## Columns

| Column | Description |
|---|---|
| `case_name` | Source document identifier (ECLI number + `.txt`), e.g. `ECLI:NL:RBOBR:2020:3584.txt` |
| `sent_text` | The sentence text |
| `label` | Gold rhetorical-role label: one of `None`, `beoordeling`, `beslissing`, `materiele feiten`, `proceshandelingen` |
| `hdr_title` | Raw section header text from the original XML, as it appeared in the source document |
| `hdr_match` | Whether `hdr_title` was successfully matched to a normalized header group |
| `hdr_group` | Normalized header group used for the header-prior Bayesian update: `Context`, `Feiten`, `Beoordeling`, `Beslissing`, or `Proceshandelingen partijen` |
| `y_labelled` | Binary indicator (0/1): 0 if `label == "None"`, 1 otherwise — the target for the None Detection (Stage 1) formulation |

## Label distribution (full dataset, 8,391 sentences)

| Label | Count | Proportion |
|---|---|---|
| None | 3,060 | 36.5% |
| beoordeling | 2,293 | 27.3% |
| materiele feiten | 1,529 | 18.2% |
| proceshandelingen | 1,263 | 15.1% |
| beslissing | 246 | 2.9% |

## Notes

- **`keep_default_na=False`**: pass this to `pandas.read_csv` when loading any of these files.
  The literal string `"None"` is a real label value in this dataset, not a missing value —
  without this flag, pandas silently converts it to `NaN`.
- Raw XML source files and intermediate annotation-tool exports are not included
  in this repository. Only the final, cleaned sentence-level data used for all experiments in the
  paper is published here.
