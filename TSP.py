import math
import random

import numpy as np


class TSP:

    def __init__(self, num_cities, population_size, num_generations):
        self.num_cities = num_cities
        self.population_size = population_size
        self.num_generations = num_generations
        self.locations = self.generate_locations()
        self.distance_matrix = self.build_distance_matrix()
        self.population = self.generate_population()

    def generate_population(self):
        population = []
        for i in range(self.population_size):
            population.append(self.generate_path())
        return population

    def generate_path(self):
        city_to_visit = [*range(1, self.num_cities + 1)]
        path = random.sample(city_to_visit, self.num_cities)
        return path

    def generate_locations(self):
        locations = [self.generate_city_angle() for _ in range(self.num_generations)]
        locations_sorted = sorted(locations)
        return locations_sorted

    @staticmethod
    def distance(angle1, angle2):
        abs_diff = abs(angle1 - angle2)
        gamma = min(abs_diff, 360 - abs_diff) # choose an acute/straight angle
        gamma_radians = math.radians(gamma)
        distance = math.sqrt(2 - 2 * math.cos(gamma_radians))  # cosines theorem
        return round(distance,2)

    def build_distance_matrix(self):
        """ Creates a matrix for quick distance lookups
            Usage:
            distance_between_cities = distance_matrix[city_index_from][city_index_to]
        """

        dimension = self.num_cities
        distance_matrix = [[0 for x in range(dimension)] for y in range(dimension)]
        for row in range(self.num_cities):
            for col in range(self.num_cities):
                distance_matrix[row][col] = self.distance(self.locations[row], self.locations[col])

        return distance_matrix

    def generate_city_angle(self):
        return round(random.uniform(0,1) * 360, 2)

    def fitness_function(self, candidate):
        pass


if __name__ == "__main__":
    tsp = TSP(num_cities=10, population_size=10, num_generations=50)

    # tests
    print(tsp.locations)
    print(tsp.population[0])
    print(tsp.distance(tsp.locations[0], tsp.locations[1]))
    print(tsp.distance(45, 330))
    print(tsp.distance(0, 180))
    print(np.array(tsp.distance_matrix))
