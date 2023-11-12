"""This is a helper module for SOM nodes"""


class Node:
    def __init__(self, index, index_normalized, position):
        self.index = index
        self.index_normalized = index_normalized
        self.position = position
        self.neighbors = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
