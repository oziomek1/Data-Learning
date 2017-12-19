"UNSUPERVISED LEARNINE"

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../input/column_2C_weka.csv')
#plt.scatter(data['pelvic_radius'], data['degree_spondylolisthesis'])

plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')

data2 = data.loc[:,['degree_spondylolisthesis','pelvic_radius']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(data2)
labels = kmeans.predict(data2)
#plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = labels)
#plt.xlabel('pelvic_radius')
#plt.xlabel('degree_spondylolisthesis')

df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)

#inertia
inertia_list = np.empty(10)
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_
#plt.plot(range(0,10), inertia_list, '-o')
#plt.xlabel('Clusters')
#plt.ylabel('Inertia')
#plt.show()

data = pd.read_csv('../input/column_2C_weka.csv')
data3 = data.drop('class',axis = 1)

#Standarization
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
kmeans = KMeans(n_clusters = 2)
pipe = make_pipeline(scalar,kmeans)
pipe.fit(data3)
labels = pipe.predict(data3)
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)

#Hierarchy dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data3.loc[175:200,:], method='single')
# dendrogram(merg, leaf_rotation=90, leaf_font_size=6)


#T Disrtibuted Stochastic Neighbor Embedding T-SNE
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(data2)
x = transformed[:, 0]
y = transformed[:, 1]
color_list = ['red' if i == 'Abnormal' else 'green' for i in data.loc[:, 'class']]
#plt.scatter(x,y,c=color_list)
#plt.xlabel('pelvic_radius')
#plt.ylabel('degree_spondylolisthesis')

#PCA Principle component analysis
from sklearn.decomposition import PCA
model = PCA()
model.fit(data3)
transformed = model.transform(data3)
print('Principle components: ', model.components_)

#PCA variance
scalar = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scalar, pca)
pipeline.fit(data3)

#plt.bar(range(pca.n_components_), pca.explained_variance_)
#plt.xlabel('PCA feature')
#plt.ylabel('variance')
#plt.show()

# apply PCA
pca = PCA(n_components=2)
pca.fit(data3)
transformed = pca.transform(data3)
x = transformed[:, 0]
y = transformed[:, 1]
plt.scatter(x,y,c=color_list)
plt.show()