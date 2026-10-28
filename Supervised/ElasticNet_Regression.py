import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data Loadin' & Prep
df =(fetch_california_housing())
dataset=(pd.DataFrame(df.data))
dataset.columns=df.feature_names
print ('Dataset :\n',dataset.head())

#Independent features
x=dataset

#Dependent Features
y=df.target
print("y=",y)

#train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42) 
#X_TRAIN nd Y_TRAIN are being USED FOR TRAING THE MODEL ND X_TEST/ Y_TEST IS USED TO TEST THE MODEL

#Standardizin' the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test =scaler.transform(x_test)

#Model Callin'
from sklearn.linear_model import ElasticNet
en=ElasticNet()

#Cross-Validation
from sklearn.model_selection import GridSearchCV
prms={'alpha':[0.01,0.1,0.2,.3,.4,.5,1,2,5,10,20,30,40,50,60,70,80,90]}
enCv=GridSearchCV(en,prms,scoring='neg_mean_squared_error',cv=5)
enCv.fit(x_train,y_train)
print("Best Parameters :",enCv.best_params_)
print("Best Score:",enCv.best_score_)

#Prediction
pred=enCv.predict(x_test)
print("Prediction :\n",pred) 

#Graph PLotthin'
import seaborn as sns
sns.displot(pred-y_test, kind="kde")
plt.show()

from sklearn.metrics import r2_score
src=r2_score(y_test,pred)
print('Score =',src)