import pandas as pd 
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data Loadin' 
df=load_iris()
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names
print('Dataset :\n',dataset.head())

#independent feature
x=dataset

#dependent Feature 
y=df.target
print("y=",y)

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test=train_test_split(x,y, test_size=0.30, random_state=50)

#standardizing dataset
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#model calling
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()

#cross-validation
from sklearn.model_selection import cross_val_score
gnb.fit(x_train,y_train)
cvs =cross_val_score(gnb,x_train,y_train,scoring='accuracy',cv=10)
z=np.mean(cvs)
print(z)

#predicition
pred=gnb.predict(x_test)
print("predection\n",pred)

#graph plottin'
import seaborn as sns 
sns.displot(pred-y_test,kind="kde")
plt.show()

from sklearn.metrics import accuracy_score,classification_report
scr=accuracy_score(y_test,pred)
print(scr)