import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
df=fetch_california_housing()
print(df)
x=pd.DataFrame(df.data, columns=df.feature_names) # Fixed: removed syntax error 'ds.' and corrected column assignment
y=df.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=42)
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)
pred=dtr.predict(x_test)
print(pred)
from sklearn.metrics import r2_score
scr=r2_score(y_test, pred) # Fixed: swapped arguments to (y_true, y_pred)
prms={
    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
    'splitter':['best','random'],
    'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12],
    'max_features':['auto','sqrt','log2'] # Fixed: changed 'max_feature' to 'max_features'
}
dtr=DecisionTreeRegressor()
from sklearn.model_selection import GridSearchCV
cv=GridSearchCV(dtr,param_grid=prms,cv=5,scoring='neg_mean_squared_error') # Fixed: changed scoring to 'neg_mean_squared_error'
cv.fit(x_train,y_train)
print(cv.best_params_)
pred=cv.predict(x_test)
print(pred)
print(r2_score(y_test, pred)) # Fixed: swapped arguments to (y_true, y_pred)