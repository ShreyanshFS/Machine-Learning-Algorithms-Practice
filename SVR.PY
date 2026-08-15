import pandas as pd
import numpy as np
from sklearn.datasets import _california_housing
import matplotlib.pyplot as plt

#Dataset Loading
df=_california_housing()
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names
print("Dataset:\n",dataset.head())