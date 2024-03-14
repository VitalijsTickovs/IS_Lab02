import math
import random as r
import numpy as np

from EvolutionaryAlg import EvolutionaryAlgorithm


class CityPath:

    def __init__(self, path, distance_matrix):
        self.path = path
        self.distance_matrix = np.array(distance_matrix)

    def get_path_distance(self):
        distance = 0
        for i in range(len(self.path) - 1):
            distance += self.distance_matrix[self.path[i] - 1][self.path[i + 1] - 1]

        distance += self.distance_matrix[self.path[-1] - 1][self.path[0]-1]

        return distance

    def __len__(self):
        return len(self.path)


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
            population.append(CityPath(path=self.generate_path(),
                                       distance_matrix=self.distance_matrix))
        return population

    def generate_path(self):
        city_to_visit = [*range(1, self.num_cities + 1)]
        path = r.sample(city_to_visit, self.num_cities)
        return path

    def generate_locations(self):
        locations = [self.generate_city_angle() for _ in range(self.num_cities)]
        locations_sorted = sorted(locations)
        return locations_sorted

    @staticmethod
    def generate_city_angle():
        return round(r.uniform(0, 1) * 360, 2)

    @staticmethod
    def distance(angle1, angle2):
        abs_diff = abs(angle1 - angle2)
        gamma = min(abs_diff, 360 - abs_diff)  # choose an acute/straight angle
        gamma_radians = math.radians(gamma)
        distance = math.sqrt(2 - 2 * math.cos(gamma_radians))  # cosines theorem
        return round(distance, 2)

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

    @staticmethod
    def fitness_function(city_path):
        fitness = city_path.get_path_distance()
        return fitness

    @staticmethod
    def partially_mapped_crossover(parent1, parent2):
        cutpoints = r.randint(0, len(parent1) - 1), r.randint(0, len(parent2) - 1)
        cutpoint1 = min(cutpoints)
        cutpoint2 = max(cutpoints)

        p1 = parent1.path
        p2 = parent2.path

        size = len(parent1)
        child1_path = [None] * cutpoint1 + p2[cutpoint1:cutpoint2 + 1] + [None] * (size - cutpoint2 - 1)
        child2_path = [None] * cutpoint1 + p1[cutpoint1:cutpoint2 + 1] + [None] * (size - cutpoint2 - 1)

        for i in range(size):
            if not (cutpoint1 <= i <= cutpoint2):
                # child 1: copy bits without conflicts, the rest is substituted with mappings
                if not p1[i] in child1_path:
                    child1_path[i] = p1[i]
                else:
                    chosen_bit = p1[i]
                    while chosen_bit in child1_path:
                        chosen_bit = p1[p2.index(chosen_bit)]
                    child1_path[i] = chosen_bit

                # do the same for child 2
                if not p2[i] in child2_path:
                    child2_path[i] = p2[i]
                else:
                    chosen_bit = p2[i]
                    while chosen_bit in child2_path:
                        chosen_bit = p2[p1.index(chosen_bit)]
                    child2_path[i] = chosen_bit

        dm = parent1.distance_matrix
        return [CityPath(child1_path, dm), CityPath(child2_path, dm)]

    @staticmethod
    def swap_mutation(child):
        swap_pts = r.randint(0, len(child) - 1), r.randint(0, len(child) - 1)
        child_p = child.path
        child_p[swap_pts[0]], child_p[swap_pts[1]] = child_p[swap_pts[1]], child_p[swap_pts[0]]
        child.path = child_p
        return child

    def solve(self):
        genetic_algo = EvolutionaryAlgorithm(population=self.population, fitness_function=self.fitness_function, maximization=False)
        genetic_algo.custom_crossover = TSP.partially_mapped_crossover
        genetic_algo.custom_mutation = TSP.swap_mutation
        best_path = genetic_algo.evolve(self.num_generations)
        print(f"Best score: {genetic_algo.best_score}")
        print(
            f"Best CityPath \ndistance: {round(best_path.get_path_distance(),2)}"
            f"\nbest_path.path: {best_path.path}")
        print([self.locations[i-1] for i in best_path.path])

        optimal_distance = 0
        for i in range(len(self.locations) - 1):
            optimal_distance += self.distance_matrix[i][i+1]
        optimal_distance += self.distance_matrix[-1][0]
        print(f"Optimal path: {optimal_distance}")


if __name__ == "__main__":
    r.seed(42)

    # parameters
    num_cities = 100
    population_size = 1000
    num_generations = 1000

    tsp = TSP(num_cities=num_cities, population_size=population_size, num_generations=num_generations)

    # run
    tsp.solve()

    # tests
    # print(tsp.locations)
    # print(tsp.population[0])
    # print(tsp.distance(tsp.locations[0], tsp.locations[1]))
    # print(tsp.distance(45, 330))
    # print(tsp.distance(0, 180))
    # print(np.array(tsp.distance_matrix))
    # print(tsp.fitness_function(tsp.population[0]))
    #
    # parent1 = CityPath([3, 4, 8, 2, 7, 1, 6, 5], [])
    # parent2 = CityPath([4, 2, 5, 1, 6, 8, 3, 7], [])
    # child = CityPath([1,2,3,4,5],[])
    #
    # c1, c2 = tsp.partially_mapped_crossover(parent1, parent2)
    # print(c1.path)
    # print(c2.path)
    # print(f'c1: {c1.path}')
    # print(f'c2: {c2.path}')
    # print(tsp.swap_mutation(child).path)


