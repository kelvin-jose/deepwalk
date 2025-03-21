import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from deepwalk3 import DeepWalk

# constants
DATASET_LOC = './datasets/cora'
SAMPLE_SIZE = 50000
RANDOM_WALK_LEN = 128
EMBED_DIMS = 256
MODEL_NAME = 'myw2v.model'

cites_df = pd.read_csv(os.path.join(DATASET_LOC, "cora.cites"), sep='\t', header=None, names=["target", "source"])
content_df = pd.read_csv(os.path.join(DATASET_LOC, "cora.content"), sep='\t', header=None)[[0, 1434]]
content_df.rename({0: 'node', 1434: 'type'}, axis = 1, inplace = True)

# generating networkx graph from edgelist
graph = nx.from_pandas_edgelist(cites_df)

# model training
dw = DeepWalk(graph)
print('[x] generating training samples ...')
trainX = dw.generate_train_samples(SAMPLE_SIZE, RANDOM_WALK_LEN)
print('[x] model training ...')
model = dw.train(trainX, EMBED_DIMS)
print('[x] saving trained model ...')
model.save(MODEL_NAME)

k2i = pd.DataFrame({'node': model.wv.key_to_index.keys(), 'vindex': model.wv.key_to_index.values()})
meta_df = pd.merge(content_df, k2i, on = 'node').sort_values('vindex')

# converting embeddings to 2 dims for plotting 
pca = PCA(n_components=2)
x = pca.fit_transform(model.wv.vectors)

meta_df['pca_x'] = x[:, 0]
meta_df['pca_y'] = x[:, 1]
meta_df['type_int'] = LabelEncoder().fit_transform(meta_df['type'])

plt.scatter(meta_df['pca_x'], meta_df['pca_y'], c=meta_df['type_int'])
plt.show()