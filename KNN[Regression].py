import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt 

#DataSet Loadin'
df=(fetch_california_housing())
dataset=(pd.DataFrame(df.data))
dataset.columns=df.feature_names
print('Dataset:\n',dataset.head())

#Independent Feature
x=dataset

#Dependent Feature 
y=df.target
print("y=",y)

#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=.30,random_state=42)

#standardizin' theh dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#model callin'
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=10)

#cross-validation
from sklearn.model_selection import cross_val_score
knn.fit(x_train,y_train)
cvs=cross_val_score(knn,x_train,y_train,scoring='neg_mean_squared_error',cv=10)
z=np.mean(cvs)
print (z)

#Predection
pred=knn.predict(x_test)
print('predection\n',pred)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, pred)
mse = mean_squared_error(y_test, pred)
print("R²:", r2)
print("MSE:", mse)

#Graph Plotting'
import seaborn as sns
sns.displot(pred-y_test, kind="kde")
plt.show()

#Using Confusion metrics 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
