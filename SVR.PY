import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

#Dataset Loading
df=fetch_california_housing()
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names
print("Dataset:\n",dataset.head())

#independent Feature 
x=dataset

#dependent feature
y=df.target
print("y=",y)

#tain nd test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=50)

#standardlizing data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#model calling 
from sklearn.svm import SVR
svr= SVR()

#cross-validation
from sklearn.model_selection import cross_val_score
svr.fit(x_train,y_train)
cvs=cross_val_score(svr,x_train,y_train,cv=10,scoring='r2')
print("mean:",np.mean(cvs))

#predection
p=svr.predict(x_test)
print("Predection\n",p)

#evalution
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("R2_score:",r2_score(y_test,p),"\n","MSE:",mean_squared_error(y_test,p),"\n","MAE:",mean_absolute_error(y_test,p))

#Graph Plotting
import seaborn as sns
sns.displot(p-y_test,kind='kde')
plt.xlabel("Residual")
plt.title("Residual Distribution")
plt.show()