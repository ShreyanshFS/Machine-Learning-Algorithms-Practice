import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.datasets import load_iris
import seaborn as sns

d=load_iris()
df= sns.load_dataset('iris')
df.head()

#Independent Features
x=df.iloc[:,:-1]

#Dependent Features
y=d.target

print(x,y)

#Train-Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_tain,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
print(x_train)

#Model Callin'
from sklearn.tree import DecisionTreeClassifier
tm=DecisionTreeClassifier(max_depth=5) #aise defult case mein yeh post-prunning ho jata hai 

#Data Fittin'
tm.fit(x_train,y_tain)

#Plotting
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(tm,filled=True)
plt.show()

#Predection
y_pred=tm.predict(x_test)
print(y_pred)

#Scorin'
from sklearn.metrics import accuracy_score, classification_report
ac=accuracy_score(y_pred,y_test)
print("Accuracy score :",ac)
cr=classification_report(y_pred,y_test)
print("Classification Report :\n",cr)