import random as rn
import numpy as np
from numpy.random import choice as np_choice

class ACOm(object):

    def __init__(self, distance, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):

        self.distance  = distance
        self.pheromone = np.ones(self.distance.shape) / len(distance)
        self.all_inds = range(len(distance))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            print (shortest_path)
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone * self.decay
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 2.5 / self.distance[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distance[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distance) - 1):
            move = self.pick_move(self.pheromone[prev], self.distance[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))
        return path

    def pick_move(self, pheromone, distance, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0


        row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta) [Transversing function]
        """row1 = (dist - self.beta)**2
        row2 = pheromone(dist ** 2 - self.alpha) **2
        row = row1 + row2"""

        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move