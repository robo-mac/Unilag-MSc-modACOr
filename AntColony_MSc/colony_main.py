import numpy as randnp
import random as rd
from numpy.random import choice as np_choice

class AntCol(object):

    def __init__(self, distance, decay_rate, num_ants,  num_best, num_iterations, alpha=1, beta=1):
        #Please refer to the Arg Library below the code to understand what this variables do
        self.distance = distance
        self.pheromone = randnp.ones(self.distance.shape) / len(distance)
        self.all_ends = range(len(distance))
        self.num_ants = num_ants
        self.num_best = num_best
        self.num_iterations = num_iterations
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        overall_shortest_path = ("", randnp.inf)
        for i in range (self.num_iterations):
            all_paths = self.generate_all_paths()
            self.spread_pheromone(all_paths, self.num_ants, shortest_path = shortest_path)
            shortest_path = min(all_paths, key=lambda z: z[1])
            print (shortest_path)
            if shortest_path[1] < overall_shortest_path[1]:
                overall_shortest_path = shortest_path
            self.pheromone * self.decay_rate
        return overall_shortest_path

    def spread_pheromone (self, all_paths, num_best, shortest_path):
        sorted_path = sorted(all_paths, key=lambda z: z[1])
        for current_path, dist in sorted_path[:num_best]:
            for movement in current_path:
                self.pheromone[movement] += 1.0 / self.distance[movement]

    def generate_path_distance (self, path):
        total_distance = 0
        for element in current_path:
            total_distance += self.distance[element]
        return total_distance

    def generate_all_paths(self):
        all_paths = []
        for i in range(self.num_ants):
            current_path = self.generate_path(0)
            all_paths.append((current_path, self.generate_path_distance(current_path)))
        return all_paths

    def generate_path (self, start):
        current_path = []
        visited_node = set()
        visited_node.add(start)
        prev = start
        for i in range (len(self.distance) - 1):
            movement = self.pick_move(self.pheromone[prev], self.distance[prev], visited_node)
            current_path.append(prev, movement)
            prev = movement
            visited_node.add(movement)
        current_path.append(prev, start)
        return current_path

    def pick_move(self, pheromone, dist, visited_node):
        pheromone = np.copy(pheromone)
        pheromone[list(visited_node)] = 0

        row = pheromone ** self.alpha *((1.0 / dist) ** self.beta)

        normal_row = row / row.sum()
        movement = np_choice(self.all_ends, 1, p = normal_row)[0]
        return movement



