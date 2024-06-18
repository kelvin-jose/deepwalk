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