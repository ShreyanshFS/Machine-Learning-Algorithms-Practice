import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
df=fetch_california_housing()
print(df)
ds.
x=pd.DataFrame(df.data,ds.columns=df.features_names)
y=df.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=42)
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)
pred=dtr.predict(x_test)
print(pred)
from sklearn.metrics import r2_score
scr=r2_score(pred,y_test)
prms={
    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
    'splitter':['best','random'],
    'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12],
    'max_feature':['auto','sqrt','log2']
}
dtr=DecisionTreeRegressor()
from sklearn.model_selection import GridSearchCV
cv=GridSearchCV(dtr,param_grid=prms,cv=5,scoring='neg_mean_squared')
cv.fit(x_train,y_train)
print(cv.best_params_)
pred=cv.predict(x_test)
print(pred)
print(r2_score(pred,y_test))