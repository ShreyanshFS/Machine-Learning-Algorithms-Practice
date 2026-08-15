import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

#Dataset Loadin'
df=load_breast_cancer()
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names
print("Dataset:\n",dataset.head())

#independent Feature 
x=dataset 

#Dependent Feature 
y=df.target
print("y=",y)

#Train Test Split 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=50)

#Standardlizing Data
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#model calling
from sklearn.svm import SVC
svc=SVC()

#Cross-Validation
from sklearn.model_selection import cross_val_score
svc.fit(x_train,y_train)
cvs=cross_val_score(svc,x_train,y_train,cv=10)
m=np.mean(cvs)
print(m)

#Predection
p=svc.predict(x_test)
print("Predection\n",p)

#Evalution
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,ConfusionMatrixDisplay
print("Classifiaction Report :\n",classification_report(y_test,p))
print("Accuracy Score:\n",accuracy_score(y_test,p))
cm= confusion_matrix(y_test,p)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Graph Plotting
import seaborn as sns
sns.displot(p-y_test,kind='kde')
plt.xlabel("Residual")
plt.title("Residual Distribution")
plt.show()


