# Machine Learning Implementation & Learning Log

This folder is a practical record of learning supervised machine learning with Python. The scripts explore the complete workflow: loading data, separating features and targets, train/test splitting, scaling, fitting, validation, evaluation, and visualization.

The code is educational and experimental rather than a packaged library. Most examples use datasets bundled with scikit-learn, so no dataset files are required.

## Environment

- Python 3.8+
- NumPy, pandas, scikit-learn, Matplotlib, and Seaborn
- Git and `pip`

Install dependencies from this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Some scikit-learn datasets may be downloaded the first time they are used.

## What is implemented

### Regression

| File | Algorithm | Dataset | Main ideas |
|---|---|---|---|
| `Linear_Regression.py` | Linear Regression | California Housing | scaling, cross-validation, residual KDE, R² |
| `Rigde_Regression.py` | Ridge Regression | California Housing | L2 regularization and alpha search |
| `Lasso_Regression.py` | Lasso Regression | California Housing | L1 regularization and alpha search |
| `ElasticNet_Regression.py` | ElasticNet | California Housing | combined L1/L2 regularization |
| `KNN[Regression].py` | KNN regression | California Housing | scaling, 10-fold CV, R² and MSE |
| `SVR.PY` | Support Vector Regression | California Housing | scaling, CV, MAE/MSE/R² |
| `Decision_Tree(Regression).py` | Decision Tree Regressor | California Housing | CV, R²/MSE, tree plot |
| `Decision_Tree(regresssion).py` | Tuned Decision Tree Regressor | California Housing | `GridSearchCV` and R² |
| `o.py` | Decision Tree Regressor | California Housing | MAE/MSE/RMSE, residuals, feature importance |

### Classification

| File | Algorithm | Dataset | Main ideas |
|---|---|---|---|
| `Logistic_Regression.py` | Logistic Regression | Iris, two classes | L1/L2/ElasticNet search and reports |
| `Naive_Bay's.py` | Gaussian Naive Bayes | Iris | scaling, CV, accuracy |
| `KNN[CLASSIFICATION].py` | KNN classification | Iris | scaling, CV, report, confusion matrix |
| `SVC.PY` | Support Vector Classifier | Breast Cancer | scaling, CV, reports, confusion matrix |
| `Pre-Prunning_Decision_Tree(Classifier).py` | Decision Tree Classifier | Iris | hyperparameter search as pre-pruning |
| `Post-Prunning_Decision_Tree(Classifier).py` | Decision Tree Classifier | Iris | depth-limited tree and visualization |

### From-scratch exercise

`Rndm_forest.py` implements a small classification random forest without scikit-learn’s forest estimator. It demonstrates Gini impurity, threshold splitting, weighted impurity, random feature selection, bootstrap sampling, recursive tree construction, prediction, and majority voting.

## Learning progression

The examples reinforce that scaling matters for KNN, SVM, and regularized linear models; cross-validation is more informative than one split alone; `GridSearchCV` can tune model hyperparameters; regression uses R²/MSE/MAE/RMSE while classification uses accuracy, reports, and confusion matrices; and residual/tree plots support model interpretation.

## Running examples

```powershell
python Linear_Regression.py
python Logistic_Regression.py
python "KNN[CLASSIFICATION].py"
python "Pre-Prunning_Decision_Tree(Classifier).py"
python o.py
python Rndm_forest.py
```

Many scripts open Matplotlib or Seaborn windows and wait for them to be closed.

## Current cleanup notes

This is a learning workspace, so not every file is production-ready:

- `linear_regression_manual.py` is unfinished and currently contains invalid Python syntax.
- `tempCodeRunnerFile.py` is an editor-generated temporary file.
- `Decision_Tree(Regression).py` accesses `best_estimator_` on a `cross_val_score` result; it should plot the fitted tree or use `GridSearchCV`.
- Several older scripts contain spelling inconsistencies, duplicate imports, or metric argument ordering that should be standardized later.
- Filenames retain learning-stage spellings such as `Rigde`, `Prunning`, and `Naive_Bay's`.

## Next steps

- Finish the manual linear-regression exercise.
- Refactor repeated preprocessing and evaluation into reusable functions.
- Use `random_state` consistently and add lightweight tests.
- Correct known execution issues.
- Continue with Random Forest, Gradient Boosting, clustering, dimensionality reduction, and neural-network fundamentals.

## License

No license file is currently included. Add one before distributing this repository beyond personal learning use.
