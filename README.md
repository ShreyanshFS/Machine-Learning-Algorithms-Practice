# 🤖 Machine Learning Implementation & Dev Journey

Welcome to my Machine Learning repository! This repository serves as a hands-on digital footprint of my developer journey in Machine Learning and Data Science. Here, I write clean, practical Python implementations of foundational ML algorithms, complete with data preprocessing, hyperparameter tuning, model evaluation, and visualization.

---

## 📌 Features & Key Learning Objectives

- **From-Scratch & Scikit-Learn Workflows**: Practical pipelines covering data loading, feature scaling, model fitting, and evaluation.
- **Cross-Validation & Hyperparameter Tuning**: Using `GridSearchCV` and `cross_val_score` to optimize model performance.
- **Data Visualization**: Analyzing residual distributions (`KDE` plots) and feature relationships with `Seaborn` and `Matplotlib`.
- **Standard Preprocessing**: Consistent feature standardization using `StandardScaler` and train-test splits.

---

## 🗂️ Repository Structure

```
.
├── Linear_Regression.py       # Ordinary Least Squares (OLS) Linear Regression baseline
├── Rigde_Regression.py        # Ridge Regression (L2 Regularization) with GridSearchCV
├── Lasso_Regression.py        # Lasso Regression (L1 Regularization) for feature selection
├── ElasticNet_Regression.py   # ElasticNet (Combined L1 + L2 Regularization)
├── Logistic_Regression.py    # Logistic Regression classification (Iris dataset) with GridSearch
├── Naive_Bay's.py             # Naive Bayes Classifier implementation (GaussianNB on Iris dataset)
└── residual_distribution.png  # Sample visualization of model residual distribution
```

---

## 📊 Summary of Implemented Models

| Algorithm | Model Type | Dataset / Problem | Techniques Used |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Regression | California Housing | OLS, `StandardScaler`, `cross_val_score`, R² Evaluation |
| **Ridge Regression** | Regression | California Housing | L2 Penalty, `GridSearchCV` (`alpha` tuning), Residual Analysis |
| **Lasso Regression** | Regression | California Housing | L1 Penalty, `GridSearchCV` hyperparameter search |
| **ElasticNet** | Regression | California Housing | Combined L1/L2 Penalties, `GridSearchCV` parameter grid |
| **Logistic Regression** | Classification | Iris Dataset | `saga` Solver, L1/L2/ElasticNet penalties, `GridSearchCV` |
| **Naive Bayes** | Classification | Iris Dataset | `GaussianNB`, `StandardScaler`, `cross_val_score`, Accuracy Score |

---

## 🧠 Model Breakdown: Naive Bayes Classifier (`Naive_Bay's.py`)

**Naive Bayes** is a probabilistic classification algorithm based on Bayes' Theorem with the "naive" assumption of conditional independence between features given the class label.

### Key Concepts & Workflow:
- **Bayes' Theorem**: 
  $$P(y \mid X) = \frac{P(X \mid y) \cdot P(y)}{P(X)}$$
- **Gaussian Naive Bayes (`GaussianNB`)**: Assumes continuous features follow a Gaussian (normal) distribution.
- **Preprocessing & Standardization**: Data features are standardized using `StandardScaler` prior to training.
- **Cross-Validation**: 10-fold cross-validation (`cross_val_score`) is applied to assess model generalization.
- **Evaluation**: Accuracy score, classification report, and residual KDE plot are generated to analyze predictions against true labels.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Installation & Environment Setup
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/ShreyanshFS/Machine-Learning-Algorithms-Practice.git
cd Machine-Learning-Algorithms-Practice

# Create virtual environment (optional but recommended)
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Libraries
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 4. Running an Implementation
Execute any script directly via python:
```bash
python Linear_Regression.py
python Logistic_Regression.py
python "Naive_Bay's.py"
```

---

## 📈 Visualizing Residuals & Performance

Model performance is evaluated using metrics like **R² score** for regression and **Accuracy / Classification Report** for classification. The scripts generate residual distribution plots (`KDE` plots) to assess error distribution:

```python
import seaborn as sns
sns.displot(pred - y_test, kind="kde")
```

---

## 🗺️ Developer Roadmap & Future Plans

- [x] Supervised Learning: Linear & Regularized Regressions (Ridge, Lasso, ElasticNet)
- [x] Supervised Learning: Logistic Regression with hyperparameter tuning
- [x] Supervised Learning: Naive Bayes Classification (`GaussianNB`)
- [ ] Decision Trees & Random Forests
- [ ] Support Vector Machines (SVM)
- [ ] Clustering Algorithms (K-Means, DBSCAN)
- [ ] Deep Learning Foundations (Neural Networks with PyTorch/TensorFlow)

---

## 👤 Author & Journey

Crafted as a personal log to document code implementations, experimentations, and progress in Machine Learning. 

If you find this repository helpful or want to connect over ML topics, feel free to give it a ⭐️!
