import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np

x,y=make_blobs(n_samples=1000,centers=3,n_features=2,random_state=23)
x.shape
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=.33,random_state=42)

from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
    km=KMeans(n_clusters=k,init='k-means++')
    km.fit(x_train)
    wcss.append(km.inertia_)
print(wcss)

plt.plot(range(1,11),wcss)
plt.xticks(range(1,11))
plt.xlabel("No of cluster")
plt.ylabel("WCSS")
plt.show()

km=KMeans(n_clusters=3,init='k-means++')
y_labels=km.fit_predict(x_train)
print(y_labels)
ytest_label=km.predict(x_test)
print(ytest_label)

plt.scatter(x_train[:,0],x_train[:,1],c=y_labels)
plt.show()

plt.scatter(x_test[:,0],x_test[:,1],c=ytest_label)
plt.show()

#knee locater
from kneed import KneeLocator
kl=KneeLocator(range(1,11),wcss,curve='convex',direction='decreasing')
print(kl.elbow)

from sklearn.metrics import silhouette_score
sc=[]
for k in range(2,11):
    kms=KMeans(n_clusters=k,init="k-means++")
    kms.fit(x_train)
    scr=silhouette_score(x_train,kms.labels_)
    sc.append(scr)
print(sc)

plt.plot(range(2,11),sc)
plt.xticks(range(2,11))
plt.xlabel("no of coff")
plt.ylabel("sc")
plt.show()