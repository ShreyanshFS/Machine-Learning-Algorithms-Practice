import pandas as pd
df = pd.read_csv("Anomaly~Detection/anomaly_detection_dataset.csv")
print(df.head())
import matplotlib.pyplot as plt 
plt.scatter(df.iloc[:,1],df.iloc[:,2])
plt.show()
x=df.iloc[:,1:6]
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination='auto')
clf.fit(x)
pred=clf.predict(x)
print(pred)
import numpy as np
i = np.where(pred == -1)[0]  # Get anomaly row numbers

plt.scatter(x.iloc[:, 0], x.iloc[:, 1])
plt.show()
plt.scatter(
    x.iloc[i, 0],
    x.iloc[i, 1],
    facecolors="none",
    edgecolors="red"
)
plt.show()
plt.scatter(df.iloc[:,0],df.iloc[:,1])
plt.show()
print("d")