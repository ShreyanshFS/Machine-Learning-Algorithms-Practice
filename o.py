# ==============================
# DECISION TREE REGRESSION
# ==============================

# 1. Import Libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# 2. Loading Dataset
# ==============================

df = fetch_california_housing()

X = pd.DataFrame(df.data, columns=df.feature_names)
y = df.target

print("Dataset:")
print(X.head())

print("\nTarget:")
print(y)


# ==============================
# 3. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=50
)


# ==============================
# 4. Model Calling
# ==============================

dtr = DecisionTreeRegressor(random_state=50)


# ==============================
# 5. Cross Validation
# ==============================

cvs = cross_val_score(
    dtr,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)

print("\nCross Validation Scores:")
print(cvs)

print("\nMean CV R²:")
print(np.mean(cvs))


# ==============================
# 6. Training Model
# ==============================

dtr.fit(X_train, y_train)


# ==============================
# 7. Prediction
# ==============================

p = dtr.predict(X_test)

print("\nPredictions:")
print(p)


# ==============================
# 8. Model Evaluation
# ==============================

r2 = r2_score(y_test, p)
mae = mean_absolute_error(y_test, p)
mse = mean_squared_error(y_test, p)
rmse = np.sqrt(mse)

print("\n--- Model Evaluation ---")
print("R²:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)


# ==============================
# 9. Residual Calculation
# ==============================

residuals = y_test - p


# ==============================
# 10. Actual vs Predicted Plot
# ==============================

plt.figure(figsize=(8, 5))

plt.scatter(y_test, p)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")

plt.show()


# ==============================
# 11. Residual Plot
# ==============================

plt.figure(figsize=(8, 5))

plt.scatter(p, residuals)
plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.show()


# ==============================
# 12. Residual Distribution
# ==============================

sns.displot(
    residuals,
    kind="kde"
)

plt.xlabel("Residual")
plt.title("Residual Distribution")

plt.show()


# ==============================
# 13. Feature Importance
# ==============================

plt.figure(figsize=(8, 5))

plt.barh(
    X.columns,
    dtr.feature_importances_
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance")

plt.show()