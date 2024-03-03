import math
import random


class TSP:

    def __init__(self, num_cities, population_size, num_generations):
        self.num_cities = num_cities
        self.population_size = population_size
        self.num_generations = num_generations
        self.locations = self.generate_locations()
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
        print(f"abs_angle: {abs_diff}, gamma: {gamma}")
        distance = math.sqrt(2 - 2 * math.cos(gamma_radians))  # cosines theorem
        return distance

    def generate_city_angle(self):
        return round(random.random() * 360, 2)

    def fitness_function(self, candidate):
        pass


if __name__ == "__main__":
    tsp = TSP(num_cities=10, population_size=10, num_generations=50)
    print(tsp.locations)
    print(tsp.population[0])
    print(tsp.distance(tsp.locations[0], tsp.locations[1]))
    print(tsp.distance(45, 330))
    print(tsp.distance(0, 180))
