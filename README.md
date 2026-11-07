# Datheon_Machine_Learning_-_Data_Science_Internship
# Iris Species Classification: Logistic Regression vs Decision Tree

A supervised classification project on the classic Iris dataset. It benchmarks a Logistic Regression and a Decision Tree classifier using feature scaling, an 80/20 train/test split, and standard classification metrics.

## Contents

| File | Description |
|---|---|
| `iris_classification.ipynb` | Main deliverable: the full workflow with code, outputs and plots |
| `iris.csv` | Optional. The dataset (150 rows, 4 features, 3 species). A copy is also embedded in the notebook |

## What the notebook does

1. **Load** the data with Pandas and run basic integrity checks (shape, missing values, class balance).
2. **Explore** with a Seaborn pairplot and a feature correlation heatmap.
3. **Split** the data 80/20 (120 train / 30 test), stratified by species, with `random_state=42`.
4. **Scale** features with `StandardScaler`, fit on the training set only to avoid data leakage.
5. **Train** `LogisticRegression` and `DecisionTreeClassifier` (`max_depth=3`).
6. **Evaluate** accuracy, precision, recall and F1-score (macro-averaged), plus full classification reports.
7. **Visualise** confusion-matrix heatmaps, 2D decision boundaries (petal length vs petal width), the decision tree structure and feature importances.

## Results (test set, 30 samples)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree (max_depth=3) | 0.967 | 0.970 | 0.967 | 0.967 |

Key findings:
- *Setosa* is classified perfectly by both models.
- All errors are between *versicolor* and *virginica*, the two species that overlap in feature space.
- Petal length and petal width carry almost all of the predictive signal.
- With only 30 test samples, one misclassification shifts accuracy by about 3.3 points, so the gap between the two models is within noise. Cross-validation would give a more reliable comparison.

## Requirements

- Python 3.9+
- numpy, pandas, matplotlib, seaborn, scikit-learn
- Jupyter (`notebook`, `jupyterlab` or VS Code)

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## How to run

1. Put `iris_classification.ipynb` (and optionally `iris.csv`) in the same folder.
2. Start Jupyter from that folder:
   ```bash
   jupyter notebook iris_classification.ipynb
   ```
3. Choose **Kernel → Restart & Run All**.

The load cell uses `iris.csv` if it finds it next to the notebook. Otherwise it falls back to the embedded copy of the dataset, so the notebook runs from any folder.

## Methodology notes

- **Stratified split:** each species keeps its 1/3 share in both the train and test sets.
- **Scaler fit on train only:** the test set is transformed with the training mean and standard deviation, so no test information leaks into training.
- **Macro averaging:** precision, recall and F1 weight each class equally, which suits this balanced three-class problem.
- **Decision boundary plots:** a boundary can only be drawn in 2D, so those models are retrained on petal length and petal width only. They are for visualisation; the benchmark above uses all four features.
- **Tree depth:** `max_depth=3` limits overfitting. Change it in the training cell to experiment with an unconstrained tree.

## Possible extensions

- Stratified k-fold cross-validation for more robust metrics
- Hyperparameter tuning with `GridSearchCV`
- Additional models (k-NN, SVM, Random Forest)
- The Wine dataset as an alternative benchmark
