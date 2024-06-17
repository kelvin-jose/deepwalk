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
        
    def random_walk(self, node: int, walk_length: int) -> List:
        """
        Perform a random walk starting from a given node.

        Args:
            node (int): The starting node for the random walk.
            walk_length (int): The length of the random walk.

        Returns:
            list: A list of nodes representing the random walk.
        """
        walk = [node]
        while walk_length - 1 > 0:
            neighbors = list(self.graph.neighbors(node))
            node = random.choice(neighbors)
            walk.append(node)
            walk_length = walk_length - 1

        return walk
    
    def generate_train_samples(self, num_samples: int, walk_length: int) -> List:
        """
        Generate training samples for the DeepWalk model.

        Args:
            num_samples (int): The number of training samples to generate.
            walk_length (int): The length of each random walk.

        Returns:
            List: A list of training samples, where each sample is a list of nodes.
        """
        X = []
        for sid in tqdm(range(num_samples)):
            node = random.choice(self.nodes)
            walk = self.random_walk(node, walk_length)
            X.append(walk)
        return X
    
    def train(self, X: List, embed_dim:int = 128, window:int = 5, min_count:int = 1, workers:int = 4) -> W2V:
        """
        Train the DeepWalk model using the generated training samples.

        Args:
            X (List): A list of training samples, where each sample is a list of nodes.
            embed_dim (int): The dimensionality of the word embeddings (default is 128).
            window (int): The maximum distance between the current and predicted word within a sentence (default is 5).
            min_count (int): Ignores all words with a total frequency lower than this (default is 1).
            workers (int): The number of CPU cores to use for training (default is 4).

        Returns:
            gensim.models.word2vec.Word2Vec: The trained Word2Vec model.
        """
        w2v = Word2Vec(sentences = X, 
              vector_size = embed_dim, 
              window = window, 
              min_count = min_count, 
              workers = workers)
        return w2v