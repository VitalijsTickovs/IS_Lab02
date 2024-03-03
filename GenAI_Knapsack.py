import random
import math

from EvolutionaryAlg import EvolutionaryAlgorithm


class Knapsack:
    def __init__(self, weights, values, representation, maximum_weight):
        self.weights = weights
        self.values = values
        self.maximum_weight = maximum_weight
        self.representation = representation
        self.total_weight = self.get_total_weight()
        self.total_value = self.get_total_value()

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, index):
        return Knapsack(self.weights[index], self.values[index],
                        self.representation[index],
                        self.maximum_weight)

    def __add__(self, other):
        return Knapsack(self.weights + other.weights,
                        self.values + other.values,
                        self.representation + other.representation,
                        self.maximum_weight)

    def __setitem__(self, key, value):
        self.representation[key] = value
        self.total_weight = self.get_total_weight()
        self.total_value = self.get_total_value()

    def get_total_weight(self):
        if type(self.representation) is int:
            return self.weights
        total_weight = 0
        for i in range(len(self.representation)):
            total_weight += self.representation[i] * self.weights[i]
        return total_weight

    def get_total_value(self):
        if type(self.representation) is int:
            return self.values
        total_value = 0
        for i in range(len(self.representation)):
            total_value += self.representation[i] * self.values[i]
        return total_value


class KnapsackProblem:

    def __init__(self, maximum_weight, num_items, population_size, num_generations):
        self.maximum_weight = maximum_weight
        self.num_items = num_items
        self.num_generations = num_generations
        self.population_size = population_size
        self.weights = self.generate_weights()
        self.values = self.generate_values()
        self.population = self.generate_population()

    def generate_values(self):
        values = list(range(self.num_items))
        for i in range(self.num_items):
            values[i] = random.randint(1, 5)
        return values

    def generate_weights(self):
        """ The method generates the initial weights for the items in the knapsack """
        weights = []
        for i in range(self.num_items):
            random_weight = random.random() * 10
            # saving up to 2 decimal places
            weights.append(math.floor(random_weight * 100) / 100.0)
        return weights

    def generate_knapsack_representation(self):
        representation = list(range(self.num_items))
        for i in range(self.num_items):
            representation[i] = random.randint(0, 1)
        return representation

    def generate_population(self):
        population = []
        for i in range(self.population_size):
            population.append(Knapsack(self.weights, self.values,
                                       self.generate_knapsack_representation(),
                                       self.maximum_weight))
        return population

    @staticmethod
    def fitness_function(knapsack):
        representation = knapsack.representation
        score = 0
        ratios = KnapsackProblem.count_ratios(knapsack)

        for i in representation:
            score += i * knapsack.values[i]

        total_weight_in_knapsack = knapsack.get_total_weight()

        while total_weight_in_knapsack > knapsack.maximum_weight:
            penalty_idx = KnapsackProblem.penalize(ratios)
            penalty = knapsack.values[penalty_idx]
            # multiply by ratio to strengthen the effect of penalty
            score -= penalty*round(ratios[penalty_idx])
            total_weight_in_knapsack -= knapsack.weights[penalty_idx]
            ratios[penalty_idx] = 0

        if score < 0:
            score = 0

        return score

    @staticmethod
    def penalize(ratios):
        """ Finds an index of an item in the knapsack with the highest weight/value ratio
            (meaning that the item is the least useful)
        """
        maximum = 0
        max_idx = -1
        for i in range(len(ratios)):
            if ratios[i] > maximum and ratios[i] != 0:
                maximum = ratios[i]
                max_idx = i
        return max_idx

    @staticmethod
    def count_ratios(knapsack):
        """ Counts weight/value ratios of each item in the knapsack"""
        ratios = [0]*(len(knapsack))
        for i in range(len(knapsack.representation)):
            if knapsack.representation[i] == 1:
                ratios[i] = knapsack.weights[i] / knapsack.values[i]
        return ratios

    def solve(self):
        genetic_algo = EvolutionaryAlgorithm(population=self.population, fitness_function=self.fitness_function)
        best_knapsack = genetic_algo.evolve(50)
        print(genetic_algo.scores)
        print(f"Best score: {genetic_algo.best_score}")
        print(f"Best knapsack\nweight: {round(best_knapsack.get_total_weight())}\nvalue: {best_knapsack.get_total_value()}")


if __name__ == "__main__":
    knapsack_problem = KnapsackProblem(maximum_weight=100,
                                       num_items=100,
                                       population_size=100,
                                       num_generations=50)
    knapsack_problem.solve()
