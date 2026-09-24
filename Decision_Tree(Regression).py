from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

#Loadin' Dataset
df=fetch_california_housing()
ds=pd.DataFrame(df.data)
ds.columns=df.feature_names
print('Dataset:\n',ds.head())

#Independent Feature
x=ds
#Dependent Feature
y=df.target
print('y=',y)

#train test split 
from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=.33,random_state=50)

#model callin'
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()

#Cross-Validation
from sklearn.model_selection import cross_val_score
dtr.fit(x_train,y_train)
cvs=cross_val_score(dtr,x_train,y_train,scoring="r2")
z=np.mean(cvs)
print (z)

#Predection
p=dtr.predict(x_test)
print("Predection\n",p)

#Scoring
from sklearn.metrics import r2_score, mean_squared_error
print("R²:", r2_score(y_test,p))
print("MSE:",mean_squared_error(y_test,p))

#Graph Plotting
#sns.displot(p-y_test,kind="kde")
#plt.show()
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(cvs.best_estimator_, filled=True)
plt.show()
