import random
import math

from EvolutionaryAlg import EvolutionaryAlgorithm

class Knapsacks:
    def __init__(self, n=5):
        self.generation = 0
        self.maximum_weight = 5*10
        self.weights = self.generate_weights(n)
        self.value = self.generate_value(n)
        self.population = self.generate_knapsacks(n)
    def generate_value(self, n):
        population = list(range(n))
        for i in range(n):
            population[i] = random.randint(1, 5)
        return population

    def generate_knapsacks(self, n):
        population = []
        for i in range(n):
            population.append(self.generate_knapsack(n))
        return population

    def generate_knapsack(self, n):
        population = list(range(n))
        for i in range(n):
            population[i] = random.randint(0, 1)
        return population

    def generate_weights(self, n):
        """
            The method generates the initial weights for the items in the knapsack

            input: n - number of items in the knapsack
        """
        weights = []
        for i in range(n):
            random_weight = random.random() * 10
            # saving up to 2 decimal places
            weights.append(math.floor(random_weight * 100) / 100.0)
        return weights

    def fitness_function(self, population):
        score = 0
        ratio = self.count_ratio(population)
        for i in population:
            score += i * self.value[i]

        total_weight_in_knapsack = 0
        for i in population:
            total_weight_in_knapsack += population[i] * self.weights[i]

        while total_weight_in_knapsack > self.maximum_weight:
            penalty_idx = self.penalize(ratio)
            penalty = self.value[penalty_idx]
            ratio.remove(penalty_idx)
            score -= penalty

        return score

    def penalize(self, ratio):
        minimum = math.inf
        minimum_idx = -1
        for i in range(len(ratio)):
            if ratio[i] < minimum and ratio[i] != 0:
                minimum = ratio[i]
                minimum_idx = i
        return minimum_idx

    def count_ratio(self, population):
        total_score = 0
        ratio = list(range(len(population)))
        for i in population:
            if i == 1:
                ratio[i] = self.weights[i] / self.value[i]
        return ratio

    def main_loop(self):
        genetic_algo = EvolutionaryAlgorithm(population=self.population, fitness_function=self.fitness_function)
        genetic_algo.evolve(50)
        print(genetic_algo.scores)


if __name__ == "__main__":
    kp = Knapsacks()
    kp.main_loop()
