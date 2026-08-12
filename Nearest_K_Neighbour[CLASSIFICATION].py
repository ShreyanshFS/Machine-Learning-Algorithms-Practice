import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

#Dataset Loadin'
df= load_iris()
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names
print("Dataset :\n",dataset.head())

#independet Feature
x=dataset

#Dependent Feature
y=df.target
print("y=",y)

#Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_traim,y_test=train_test_split(x,y,test_size=0.30,random_state=50)

#Standardlizing Data
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test) 

#model callin' 
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)

#cross-validation
from sklearn.model_selection import cross_val_score
knn.fit(x_train,y_traim)
cvs=cross_val_score(knn,x_train,y_traim,scoring='accuracy',cv=10)
z=np.mean(cvs)
print (z)

#Predection
pred=knn.predict(x_test)
print('predection\n',pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
from sklearn.metrics import accuracy_score,classification_report
src= accuracy_score(y_test,pred)
print (src)

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
