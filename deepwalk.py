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