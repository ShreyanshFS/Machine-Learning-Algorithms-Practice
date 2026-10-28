import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

ir=load_iris()
data=pd.DataFrame(ir.data)
data.columns=ir.feature_names
print(data)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_sc=ss.fit_transform(data)
print(x_sc)
x_sc.shape

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca_ss=pca.fit_transform(x_sc)
plt.scatter(pca_ss[:,0],pca_ss[:,1])
plt.show()

import scipy.cluster.hierarchy as hc
plt.figure(figsize=(20,7))
plt.title("Dendograms")
hc.dendrogram(hc.linkage(pca_ss,method='ward'))
plt.title('Dendograms')
plt.xlabel('Sample Index')
plt.ylabel('Eucledian Index')
plt.show()

from sklearn.cluster import AgglomerativeClustering
c=AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='ward')
c.fit(pca_ss)
c.labels_
plt.scatter(pca_ss[:,0],pca_ss[:,1],c=c.labels_)
plt.show()

from sklearn.metrics import silhouette_score
sc=[]
for k in range(2,11):
    agg=AgglomerativeClustering(n_clusters=k,metric='euclidean',linkage='ward')
    agg.fit(x_sc)
    scr=silhouette_score(x_sc,agg.labels_)
    sc.append(scr)
print(sc)