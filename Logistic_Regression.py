import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


df=sns.load_dataset('iris')
print(df.head())

print(df['species'].unique())
print(df.isnull().sum())
print(df['species']!='setosa')
df=df[df['species']!='setosa']
df.head()
print(df.head())

df['species']=df['species'].map({'versicolor':0,'virginica':1})
df.head()
print(df.head())

#Split Dataset into independent nd dependent features
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
print(x,y)

#train test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=.25,random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#model calling
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='saga',max_iter=5000)

#cross validation 
from sklearn.model_selection import GridSearchCV
prms=[
    {
        'penalty':['l1'],
        'C':[1,2,3,4,5,6,10,20,30,40,50],
        'max_iter':[5000]
    },
    {
        'penalty':['l2'],
        'C':[1,2,3,4,5,6,10,20,30,40,50],
        'max_iter':[5000]
    },
    {
        'penalty':['elasticnet'],
        'C':[1,2,3,4,5,6,10,20,30,40,50],
        'max_iter':[5000],
        'l1_ratio':[0.0,0.5,1.0]
    }
]
c_reg=GridSearchCV(lr,param_grid=prms,scoring='accuracy',cv=5)

c_reg.fit(x_train,y_train)
print(c_reg.best_params_)
print(c_reg.best_score_)

#predection
pred=c_reg.predict(x_test)

#accuracy
from sklearn.metrics import accuracy_score,classification_report
scr=accuracy_score(y_test,pred)
print(scr)

print(classification_report(y_test,pred))

#eda
sns.pairplot(df,hue='species')
plt.show()

print(df.corr())  