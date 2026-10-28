import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns

#Loadin' data
iris = load_iris()
df=sns.load_dataset('iris')
df.head()

#Independent Feature
x=df.iloc[:,:-1]

#Dependent Feature
y=iris.target
print(x,y)

#Train-Test Split 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=42)
print(x_train)

#Model calling
from sklearn.tree import DecisionTreeClassifier
tm=DecisionTreeClassifier()

#Cross-validation
from sklearn.model_selection import GridSearchCV
prms={
    'criterion':['gini','entropy','log_loss'],
    'splitter':['best','random'],
    'max_depth':[1,2,3,4,5],
    'max_features':['sqrt','log2']
}
cv=GridSearchCV(tm,param_grid=prms,cv=5,scoring='accuracy')
cv.fit(x_train,y_train)
print("best prms :\n",cv.best_params_)

#prediction
y_pred=cv.predict(x_test) 
print(y_test,"\n",y_pred)

#Scorin'
from sklearn.metrics import accuracy_score, classification_report
scr=accuracy_score(y_pred,y_test)
print("Acurracy Score",scr)
print("Classification Report \n",classification_report(y_pred,y_test))

#Plottin'
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(cv.best_estimator_, filled=True)
plt.show()
