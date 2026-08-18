# 🤖 Machine Learning Implementation & Dev Journey

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Repo Stars](https://img.shields.io/github/stars/ShreyanshFS/Machine-Learning-Algorithms-Practice?style=social)

Welcome to my Machine Learning repository! This repository serves as a hands-on digital footprint of my developer journey in Machine Learning and Data Science. Here, I write clean, practical Python implementations of foundational ML algorithms, complete with data preprocessing, hyperparameter tuning, model evaluation, and visualization.

---

## 📌 Features & Key Learning Objectives

- **From-Scratch & Scikit-Learn Workflows**: Practical pipelines covering data loading, feature scaling, model fitting, and evaluation.
- **Cross-Validation & Hyperparameter Tuning**: Using `GridSearchCV` and `cross_val_score` to evaluate and optimize model performance.
- **Data Visualization**: Analyzing residual distributions (`KDE` plots), feature relationships, and confusion matrices with `Seaborn` and `Matplotlib`.
- **Standard Preprocessing**: Consistent feature standardization using `StandardScaler` and train-test splits.

---

## 🗂️ Repository Structure

```
.
├── Linear_Regression.py       # Ordinary Least Squares (OLS) Linear Regression baseline
├── Rigde_Regression.py        # Ridge Regression (L2 Regularization) with GridSearchCV
├── Lasso_Regression.py        # Lasso Regression (L1 Regularization) for feature selection
├── ElasticNet_Regression.py   # ElasticNet (Combined L1 + L2 Regularization)
├── Logistic_Regression.py     # Logistic Regression classification (Iris dataset) with GridSearch
├── Naive_Bay's.py             # Naive Bayes Classifier implementation (GaussianNB on Iris dataset)
├── KNN[CLASSIFICATION].py     # K-Nearest Neighbors Classification (Iris dataset)
├── KNN[Regression].py         # K-Nearest Neighbors Regression (California Housing dataset)
├── SVC.PY                     # Support Vector Classifier (Breast Cancer dataset)
├── SVR.PY                     # Support Vector Regressor (California Housing dataset)
├── residual_distribution.png  # Sample visualization of model residual distribution
└── requirements.txt           # Python dependencies list
```

---

## 📊 Summary of Implemented Models

| Algorithm | Model Type | Dataset / Problem | Key Techniques & Metrics |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Regression | California Housing | OLS, `StandardScaler`, `cross_val_score`, R² Evaluation |
| **Ridge Regression** | Regression | California Housing | L2 Penalty, `GridSearchCV` (`alpha` tuning), Residual Analysis |
| **Lasso Regression** | Regression | California Housing | L1 Penalty, `GridSearchCV` hyperparameter search |
| **ElasticNet** | Regression | California Housing | Combined L1/L2 Penalties, `GridSearchCV` parameter grid |
| **Logistic Regression** | Classification | Iris Dataset | `saga` Solver, L1/L2/ElasticNet penalties, `GridSearchCV` |
| **Naive Bayes** | Classification | Iris Dataset | `GaussianNB`, `StandardScaler`, `cross_val_score`, Accuracy Score |
| **KNN Classifier** | Classification | Iris Dataset | `KNeighborsClassifier`, `StandardScaler`, 10-fold CV, Confusion Matrix |
| **KNN Regressor** | Regression | California Housing | `KNeighborsRegressor`, `StandardScaler`, 10-fold CV, R² & MSE Evaluation |
| **Support Vector Classifier (SVC)** | Classification | Breast Cancer Dataset | `SVC`, `StandardScaler`, 10-fold CV, Confusion Matrix & Classification Report |
| **Support Vector Regressor (SVR)** | Regression | California Housing | `SVR`, `StandardScaler`, 10-fold CV (R² scoring), MAE, MSE, R² Evaluation |

---

## 🧠 Model Breakdown Highlights

### 1. Regression Models (Linear, Ridge, Lasso, ElasticNet)
- **Concept**: Predicting continuous target values using linear relationships and various regularization techniques (L1/L2 penalties) to prevent overfitting.
- **Workflow**: `StandardScaler`, `GridSearchCV` for tuning `alpha`, evaluation via R² and MSE, and KDE residual distribution plots.

### 2. Logistic Regression & Naive Bayes
- **Concept**: Probabilistic classifiers for categorical target variables.
- **Workflow**: Advanced solver (`saga`) tuning in Logistic Regression for handling elasticnet penalties. Gaussian Naive Bayes assuming feature independence.

### 3. K-Nearest Neighbors (KNN)
- **Concept**: Distance-based non-parametric algorithm classifying or predicting values based on $k$ nearest data points.
- **Workflow**: Feature scaling, 10-fold CV, evaluation via accuracy / R² & MSE, and confusion matrix visualization.

### 4. Support Vector Machines (SVC & SVR)
- **Concept**: Finds optimal hyperplanes to separate classes (SVC) or fit data within an $\epsilon$-margin (SVR).
- **Workflow**: Feature standardization, 10-fold CV scoring, robust metrics including classification reports and regression error metrics.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- **Python 3.8 or higher**
- **Git**
- **pip** package manager

### 2. Installation & Environment Setup
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/ShreyanshFS/Machine-Learning-Algorithms-Practice.git
cd Machine-Learning-Algorithms-Practice

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Running an Implementation
Execute any script directly via python:
```bash
python Linear_Regression.py
python Logistic_Regression.py
python "Naive_Bay's.py"
python "KNN[CLASSIFICATION].py"
python "KNN[Regression].py"
python SVC.PY
python SVR.PY
```

---

## 📈 Visualizing Residuals & Performance

Model performance is evaluated using metrics like **R² score**, **MSE**, and **MAE** for regression, and **Accuracy / Classification Report / Confusion Matrix** for classification. The scripts generate residual distribution plots (`KDE` plots) to assess error distribution:

```python
import seaborn as sns
sns.displot(pred - y_test, kind="kde")
```

---

## 🗺️ Developer Roadmap & Future Plans

- [x] Supervised Learning: Linear & Regularized Regressions (Ridge, Lasso, ElasticNet)
- [x] Supervised Learning: Logistic Regression with hyperparameter tuning
- [x] Supervised Learning: Naive Bayes Classification (`GaussianNB`)
- [x] Supervised Learning: K-Nearest Neighbors (KNN Classification & Regression)
- [x] Supervised Learning: Support Vector Machines (`SVC` & `SVR`)
- [ ] Decision Trees & Random Forests
- [ ] Ensemble Learning (Random Forests, Gradient Boosting, XGBoost)
- [ ] Clustering Algorithms (K-Means, DBSCAN)
- [ ] Deep Learning Foundations (Neural Networks with PyTorch/TensorFlow)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software for educational and personal projects.

---

## 👤 Author & Journey

Crafted as a personal log to document code implementations, experimentations, and progress in Machine Learning. 

If you find this repository helpful or want to connect over ML topics, feel free to give it a ⭐️!
