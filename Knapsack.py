import random
import math

import numpy as np
import pulp
import pandas as pd

from EvolutionaryAlg import EvolutionaryAlgorithm


class Knapsack:
    def __init__(self, weights, values, representation, maximum_weight, penalty_factor):
        self.weights = weights
        self.values = values
        self.maximum_weight = maximum_weight
        self.representation = representation
        self.total_weight = self.get_total_weight()
        self.total_value = self.get_total_value()
        self.penalty_factor = penalty_factor

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, index):
        return Knapsack(self.weights[index], self.values[index],
                        self.representation[index],
                        self.maximum_weight,
                        self.penalty_factor)

    def __add__(self, other):
        return Knapsack(self.weights + other.weights,
                        self.values + other.values,
                        self.representation + other.representation,
                        self.maximum_weight,
                        self.penalty_factor)

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
        self.total_weight = total_weight
        return total_weight

    def get_total_value(self):
        if type(self.representation) is int:
            return self.values
        total_value = 0
        for i in range(len(self.representation)):
            total_value += self.representation[i] * self.values[i]
        self.total_value = total_value
        return total_value


class KnapsackProblem:
    def __init__(self, maximum_weight, num_items, population_size, num_generations, penalty_factor=1.0):
        self.maximum_weight = maximum_weight
        self.num_items = num_items
        self.num_generations = num_generations
        self.population_size = population_size
        self.penalty_factor = penalty_factor
        self.mutation_probability = 1

        # generate a problem
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
            random_weight = random.uniform(0.1, 1) * 10
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
                                       self.maximum_weight,
                                       self.penalty_factor))
        return population

    @staticmethod
    def fitness_function(knapsack):
        ratios = KnapsackProblem.count_ratios(knapsack)
        score = knapsack.get_total_value()
        score = KnapsackProblem.penalize(ratios, knapsack, score)

        score = max(score, 0)

        return score

    @staticmethod
    def penalize(ratios, knapsack, score):
        total_weight_in_knapsack = knapsack.get_total_weight()
        while total_weight_in_knapsack > knapsack.maximum_weight:
            penalty_idx = KnapsackProblem.get_idx_max(ratios)
            penalty = knapsack.values[penalty_idx]
            # multiply by ratio to strengthen the effect of penalty
            score -= knapsack.penalty_factor * penalty
            total_weight_in_knapsack -= knapsack.weights[penalty_idx]
            ratios[penalty_idx] = 0
        # if total_weight_in_knapsack > knapsack.maximum_weight:
        #     score = 0
        return score

    @staticmethod
    def get_idx_max(ratios):
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
        ratios = [0] * (len(knapsack))
        for i in range(len(knapsack.representation)):
            if knapsack.representation[i] == 1:
                ratios[i] = knapsack.weights[i] / knapsack.values[i]
        return ratios

    def reassign_penalties(self):
        for candidate in self.population:
            candidate.penalty_factor = self.penalty_factor

    def solve(self):
        self.reassign_penalties()
        genetic_algo = EvolutionaryAlgorithm(population=self.population, fitness_function=self.fitness_function)
        genetic_algo.mutation_probability = self.mutation_probability
        genetic_algo.evolve(self.num_generations)
        # print(genetic_algo.scores)
        # print(f"Best score: {genetic_algo.best_score}")
        # print(
        #     f"Best knapsack\nweight: {round(best_knapsack.get_total_weight())}"
        #     f"\nvalue: {best_knapsack.get_total_value()}"
        #     f"\nknapsack: {best_knapsack.representation}")
        # print(f"values: {self.values}")
        # print(f"weights: {self.weights}")
        return genetic_algo.scores, genetic_algo.best_individual

    def solve_to_optimality(self):
        n = len(self.values)
        max_value = 0
        best_combination = []

        # Generate all possible combinations of items using binary representation
        for i in range(2 ** n):
            current_value = 0
            current_weight = 0
            combination = []

            # Convert i to binary representation and iterate over each bit
            for j in range(n):
                if (i >> j) & 1:
                    current_value += self.values[j]
                    current_weight += self.weights[j]
                    combination.append(j)

            # Check if the current combination is valid and update the best combination if necessary
            if current_weight <= self.maximum_weight and current_value > max_value:
                max_value = current_value
                best_combination = combination

        return max_value, best_combination

    def solve_to_optimality(self):
        # Create a binary optimization problem
        prob = pulp.LpProblem("Knapsack Problem", pulp.LpMaximize)

        # Decision variables
        x = [pulp.LpVariable(f'x{i}', cat=pulp.LpBinary) for i in range(len(self.values))]

        # Objective function: maximize total value
        prob += pulp.lpSum([self.values[i] * x[i] for i in range(len(self.values))])

        # Constraint: total weight should not exceed capacity
        prob += pulp.lpSum([self.weights[i] * x[i] for i in range(len(self.values))]) <= self.maximum_weight

        # Solve the problem
        prob.solve()

        # Extract solution
        selected_items = [i for i in range(len(self.values)) if x[i].value() == 1]
        total_value = sum([self.values[i] for i in selected_items])
        total_weight = sum([self.weights[i] for i in selected_items])
        print(selected_items)
        print(total_value, total_weight)

        return selected_items, total_value, total_weight


def find_optimal_penalty(kp, penalties):
    problem_stats = []
    i = 1
    for penalty in penalties:
        # print(f'Penalty: {penalty}')
        kp.penalty_factor = penalty
        scores, best_candidate = kp.solve()
        legal = True
        last_candidate = scores[-1][-1]
        if best_candidate.get_total_weight() > kp.maximum_weight:
            legal = False

        min_scores, avg_scores, max_scores = [s[1] for s in scores], [s[2] for s in scores], [s[3] for s in scores]

        problem_stats.append(
            [round(penalty, 2)] + [round(x, 2) for x in
                                   (min(min_scores),
                                    sum(avg_scores) / len(avg_scores),
                                    max(max_scores))
                                   ] +
            [round(last_candidate.get_total_weight(), 2),
             round(last_candidate.get_total_value(), 2),
             round(best_candidate.get_total_weight(), 2),
             round(best_candidate.get_total_value(), 2),
             legal]
        )
        # print(f'Best: {scores[-1][3]}, Average:  {scores[-1][2]}, Worst: {scores[-1][1]}')
        print(i, '/', len(penalties))
        i += 1

    stats_df = pd.DataFrame(problem_stats,
                            columns=['penalty_factor', 'min', 'average', 'max',
                                     'last_candidate_weight', 'last_candidate_value', 'best_candidate_weight',
                                     'best_candidate_value', 'best_candidate_solution_legal'])
    stats_df.to_csv(f'experiments/knapsack_maxweight-{kp.maximum_weight}_numitems-{kp.num_items}.csv', index=False)


def track_fitness_over_generations(kp):
    print('Fitness tracking, V-W:', kp.values, '\n', kp.weights)
    scores, best_candidate = kp.solve()
    value = best_candidate.get_total_value()
    weight = round(best_candidate.get_total_weight(), 2)
    bi = best_candidate

    stats = [list(x[:-1]) + [round(bi.get_total_weight(), 2),
                             round(bi.get_total_value(), 2),
                             bi.get_total_weight() <= bi.maximum_weight]
             for x in scores]

    stats_df = pd.DataFrame(stats, columns=['generation_number', 'min', 'average', 'max',
                                            'best_candidate_weight', 'best_candidate_value',
                                            'legal'])
    stats_df.to_csv(
        f'experiments/knapsack_fitness_penalty-{kp.penalty_factor}_maxW-{kp.maximum_weight}_numIt-{kp.num_items}_V-W-[{value},{weight}].csv',
        index=False)


def tune_mutation_rate(kp, mutation_probabilities):
    num_runs = 10
    mutation_stats = []
    j = 0
    populations = generate_problem_parameters(kp, num_runs)
    for mp in mutation_probabilities:
        kp.mutation_probability = mp
        running_stats = []
        legal_count = 0
        for i in range(num_runs):
            kp.population = populations[i]
            _, best_individual = kp.solve()
            if best_individual.get_total_weight() <= kp.maximum_weight:
                legal_count += 1
                running_stats.append(round(best_individual.get_total_value(), 2))

        max_value = max(running_stats)
        min_value = min(running_stats)
        avg_value = sum(running_stats) / len(running_stats)
        mutation_stats.append(
            [round(mp, 2),
             round(min_value, 2),
             round(avg_value, 2),
             round(max_value, 2),
             legal_count])
        # print(f'Best: {min_value}, Average:  {avg_value}, Worst: {max_value}')
        print(j, '/', len(mutation_probabilities))
        j += 1

    stats_df = pd.DataFrame(mutation_stats,
                            columns=['mutation_probability', 'min',
                                     'average', 'max', 'legal_reached'])
    stats_df.to_csv(f'experiments/mutation_probs_tuning__kp_maxweight-{kp.maximum_weight}_numitems-{kp.num_items}.csv',
                    index=False)


def generate_problem_parameters(kp, num_retries):
    problem_parameters = []
    for _ in range(num_retries):
        problem_parameters.append(
            kp.generate_population()
        )
    return problem_parameters


if __name__ == "__main__":
    random.seed(42)

    # parameters
    penalty_factor = 50
    max_weight = 200
    num_items = 100
    population_size = 100
    num_generations = 50
    max_penalty = int(max_weight / 2)
    mutation_probability = 0.1

    knapsack_problem = KnapsackProblem(maximum_weight=max_weight,
                                       num_items=num_items,
                                       population_size=population_size,
                                       num_generations=num_generations,
                                       penalty_factor=penalty_factor)

    knapsack_problem.mutation_probability = mutation_probability

    # penalties = np.linspace(1, max_penalty, num=10)

    # mutations
    mutation_probabilities = np.linspace(0, 1, num=10)
    tune_mutation_rate(knapsack_problem, mutation_probabilities)

    # penalties
    # find_optimal_penalty(knapsack_problem, penalties)
    # for p in [5,10,15]:
    #     knapsack_problem.penalty_factor = p
    #     track_fitness_over_generations(knapsack_problem)

    # optimal = knapsack_problem.solve_to_optimality()
    # print('Optimal: ', optimal)
    # print(knapsack_problem.values, knapsack_problem.weights)
