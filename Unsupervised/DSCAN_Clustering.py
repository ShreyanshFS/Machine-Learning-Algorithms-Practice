from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

x,y=make_moons(n_samples=250,noise=.05)
print(x)
plt.scatter(x[:,0],x[:,1])
plt.show()

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_sc=ss.fit_transform(x)

dbcan=DBSCAN(eps=0.5)
dbcan.fit(x_sc)
dbcan.labels_
plt.scatter(x[:,0],x[:,1],c=dbcan.labels_)
plt.show()
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()