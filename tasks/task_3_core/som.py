"""This is a helper module that contains SOM model definition"""
import random
import math
from itertools import chain

import numpy as np

from tasks.task_3_core.node import Node


class SOM:

    def get_neighbors_by_index(self, node):
        neighbors = []

        deltas = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1]),
        ]

        for delta in deltas:
            position = node.index + delta
            x, y = position

            if x < 0 or x >= self.shape[0]:
                continue

            if y < 0 or y >= self.shape[1]:
                continue

            neighbors.append(self.nodes[x][y])

        return neighbors

    def initialize_nodes(self):
        self.nodes = []
        for x in range(self.shape[0]):
            row = []

            for y in range(self.shape[1]):
                index = np.array((x, y))
                index_norm = index / (max(self.shape) - 1)
                position = np.pad(index_norm, (0, self.input_shape - 2), 'constant', constant_values=0)

                node = Node(index, index_norm, position)
                row.append(node)

            self.nodes.append(row)

        for row in self.nodes:
            for node in row:
                neighbors = self.get_neighbors_by_index(node)

                for neighbor in neighbors:
                    node.add_neighbor(neighbor)

        self.nodes = list(chain.from_iterable(self.nodes))

    def reset_callbacks(self):
        self.total_distance = 0
        self.erroneous_nodes_count = 0
        self.iterations = 0

    def __init__(self, shape, input_shape, a0=0.8, s0=0.5):
        self.a0 = a0
        self.s0 = s0

        self.shape = shape
        self.input_shape = input_shape

        self.initialize_nodes()

        self.total_distance = 0
        self.erroneous_nodes_count = 0
        self.iterations = 0

    @staticmethod
    def distance_function_sq(p1, p2):
        point1 = np.array(p1)
        point2 = np.array(p2)

        sum_sq = np.sum(np.square(point1 - point2))
        return sum_sq

    def distance_function(self, p1, p2):
        sum_sq = self.distance_function_sq(p1, p2)
        return np.sqrt(sum_sq)

    def alpha(self, iteration, max_iterations):
        return self.a0 * math.exp(- iteration / max_iterations)

    def sigma(self, iteration, max_iterations):
        return self.s0 * math.exp(- iteration / max_iterations)

    def neighborhood_function(self, bmu, node, iteration, max_iterations):
        """Cosine function"""
        s = self.sigma(iteration, max_iterations)
        d = self.distance_function_sq(bmu, node)

        return math.cos(math.pi * d / (2 * s)) if d < s else 0

    def train(self, training_data, max_iterations):
        self.initialize_nodes()
        self.reset_callbacks()

        while self.iterations < max_iterations:
            example = random.choice(training_data)
            nodes_sorted = sorted(self.nodes, key=lambda n: self.distance_function_sq(n.position, example))
            bmu, sbmu = nodes_sorted[:2]

            self.total_distance += self.distance_function_sq(bmu.position, example)
            if sbmu not in bmu.neighbors:
                self.erroneous_nodes_count += 1

            bmu_pos = bmu.index_normalized

            for node in self.nodes:
                node.position = node.position + (
                        self.neighborhood_function(bmu_pos, node.index_normalized, self.iterations, max_iterations)
                        * self.alpha(self.iterations, max_iterations)
                        * (example - node.position)
                )

            self.iterations += 1

    def quantization_error(self):
        return self.total_distance / self.iterations

    def topographic_error(self):
        return self.erroneous_nodes_count / self.iterations

    def score(self):
        q_error = self.quantization_error()
        t_error = self.topographic_error()

        print(f'Quantization error: {q_error}\nTopographic error: {t_error}')
