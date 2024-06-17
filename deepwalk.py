import random
from tqdm import tqdm
from gensim.models import Word2Vec

from typing import List
from networkx.classes.graph import Graph
from gensim.models.word2vec import Word2Vec as W2V

class DeepWalk:
    def __init__(self, graph: Graph):
        """
        Initialize the DeepWalk algorithm with a given graph.

        Args:
            graph (networkx.classes.graph.Graph): The input graph for the DeepWalk model.
        """
        self.graph = graph
        self.nodes = list(graph.nodes)