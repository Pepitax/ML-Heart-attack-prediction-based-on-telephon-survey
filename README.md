# CS-433 — Project 1

Binary classification on an imbalanced dataset.  
We compare several linear models (Ridge, Logistic Regression, linear SVM) with careful preprocessing, cross-validation, and threshold tuning for F1.

The notebook **run.ipynb** orchestrates preprocessing, cross-validation, model training, and evaluation. You can toggle augmentation, K-folds, and search grids in the config cell at the top.

## Repository structure

```
.
├─ DataProcessing/
│  └─ Datasets/
│     ├─ x_train_top20.csv
│     ├─ y_train_top20.csv
│     ├─ fullDataProcess.ipynb
│     └─ top22Process.ipynb
├─ README.md
├─ helpers.py
├─ implementations.py
├─ ml-project-1-partie1team-main.zip (template/archive)
├─ project1_description.pdf
├─ project1_report.pdf
└─ run.ipynb
```

- **run.ipynb** – main entry point; runs the whole pipeline end-to-end.  
- **implementations.py** – all core ML routines used by the notebook.  
- **helpers.py** – utility functions for some of the implementation.py functions.  
- **DataProcessing/** – exploratory and data-prep notebooks and input CSVs.

- **project1_description.pdf** - the problem description
- **project1_report.pdf** - our team's report 

---

## Data

- `x_train_top20.csv` – features.  
- `y_train_top20.csv` – labels in `{-1, +1}` (mapped internally to `{0,1}` where needed).

> **Imbalance note:** the dataset is skewed. We therefore monitor **F1** (and optionally ROC-AUC) and tune the **decision threshold** rather than relying on accuracy or a fixed 0.5 cutoff.

---

## Reproducing our results

Open `run.ipynb` and execute all cells. The top “Config” cell includes:
- augmentation factor (times to replicate positive samples),
- K for K-fold,
- search grids (use **logarithmic** spacing for `λ`),
- polynomial degree (for the bonus experiment),
- random seed for reproducibility.

**Outputs**
- Best hyperparameters by CV,
- Best decision threshold by F1,
- Metrics and (optionally) plots.

---

## Common pitfalls (and how we avoid them)

- **Categoricals vs. continuous**  
  Don’t impute categorical features with mean/median; use mode (or encode first).  
- **Grid search spacing**  
  Hyperparameters like `λ` should be searched on a **log scale**, not linear.  
- **Loss expects `{0,1}`**  
  Some logistic-loss implementations require `{0,1}` labels—map from `{-1,+1}` first.  
- **Pick the right metric**  
  With imbalance, **accuracy is misleading**. Optimize **F1** (or ROC-AUC) and tune the **threshold**.  
- **Monitor the target metric during training**  
  Track F1 across epochs/steps; don’t only check at the end.

---

## Notes & limitations
- The polynomial-feature ridge was only run on the reduced dataset for time reasons; running it with augmentation could further improve F1.
- Our “penalty check” is equivalent to **threshold calibration** to trade precision/recall; we keep the model fixed and only adjust the decision cutoff on validation data.
