# CS-433 — Project 1

Binary classification on an imbalanced dataset.  
We compare several linear models (Ridge, Logistic Regression, linear SVM) with careful preprocessing, cross-validation, and threshold tuning for F1.

## TL;DR (how to run)

1. **Clone / open the repo** and make sure you have Python ≥3.9.
2. Install minimal deps:
   ```bash
   pip install numpy pandas matplotlib jupyter
   ```
3. Launch the notebook and run all:
   ```bash
   jupyter notebook run.ipynb
   ```
   The notebook orchestrates preprocessing, cross-validation, model training, and evaluation. You can toggle augmentation, K-folds, and search grids in the config cell at the top.

---

## Repository structure

```
.
├─ DataProcessing/
│  └─ Datasets/
│     ├─ x_train_top20.csv
│     ├─ y_train_top20.csv
│     ├─ fullDataProcess.ipynb
│     └─ top22Process.ipynb
├─ helpers.py
├─ implementations.py
├─ run.ipynb
├─ project1_description.pdf
├─ README.md
└─ ml-project-1-partie1team-main.zip   (template/archive)
```

- **run.ipynb** – main entry point; runs the whole pipeline end-to-end.  
- **implementations.py** – all core ML routines used by the notebook.  
- **helpers.py** – utility functions.  
- **DataProcessing/** – exploratory and data-prep notebooks and input CSVs.

---

## Data

- `x_train_top20.csv` – features.  
- `y_train_top20.csv` – labels in `{-1, +1}` (mapped internally to `{0,1}` where needed).

> **Imbalance note:** the dataset is skewed. We therefore monitor **F1** (and optionally ROC-AUC) and tune the **decision threshold** rather than relying on accuracy or a fixed 0.5 cutoff.

---

## Methodology

### 1) Preprocessing
- Drop obviously non-predictive or ID-like fields (e.g., phone, response month).
- Handle missing values (constant imputation) and **scale features** (normalization/standardization) via `normalize_and_fill(...)`.
- Map labels to `{0,1}` when a loss requires it (e.g., logistic loss).

### 2) Train/validation split & augmentation
- Split into **80% train / 20% test** (held-out).
- Optional **positive-class augmentation**: duplicate minority samples *k* times to mitigate imbalance (configurable in the notebook).

### 3) Models & selection
- **Ridge Regression**  
  - K-fold cross-validation over `λ` on a **log-scaled** grid.  
  - After fitting, sweep the **classification threshold** to maximize validation **F1** (“penalty/threshold check”).

- **Ridge + Polynomial Features (bonus)**  
  - Kernelized intuition applied explicitly via polynomial expansion.  
  - Performed on the reduced dataset (fastest and gave our best score in our tests).

- **Logistic Regression (gradient) & linear SVM (gradient)**  
  - Trained on the reduced data for speed.  
  - Same threshold-tuning procedure as above.

### 4) Evaluation
- Report **F1** on the validation folds and test split.  
- Also show accuracy/ROC-AUC for context, but avoid drawing conclusions from accuracy alone.

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

## Results (summary)

> Replace the placeholders below with your final numbers once you run the notebook.

| Model                              | Data used     | Best λ / params | Threshold | F1 (val) | F1 (test) |
|-----------------------------------|---------------|-----------------|-----------|----------|-----------|
| Ridge (linear)                    | reduced/aug?  | …               | …         | …        | …         |
| Logistic Regression (gradient)    | reduced       | …               | …         | …        | …         |
| Linear SVM (gradient)             | reduced       | …               | …         | …        | …         |
| **Ridge + polynomial features**   | reduced       | … / degree=…    | …         | …        | **…**     |

**Observation:** On this dataset, training longer or over-optimizing for accuracy tended to **hurt F1**, so we select models by **validation F1** and tune thresholds accordingly.

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

---

## Contributors
- Team: *add names here*

## License
Add your course/assignment policy or license here (if applicable).
