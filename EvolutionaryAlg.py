import math
import random
import numpy as np


class EvolutionaryAlgorithm:

    def __init__(self, population, fitness_function, maximization = True):
        self.population = population
        self.fitness_function = fitness_function
        self.scores = []

        self.maximization = maximization # True - maximization, False - minimization
        if maximization:
            self.best_score = float("-inf")
        else:
            self.best_score = float("inf")
        self.best_individual = self.population[0]
        self.crossover_probability = 1
        self.mutation_probability = 0.1
        self.custom_crossover = None
        self.custom_mutation = None

        # available: ['roulette', 'tournament', 'exp_rank']
        self.selection_type = 'roulette'

    def evolve(self, generations, print_delay=100):
        for iteration in range(generations):
            fitness_scores = [self.fitness_function(candidate) for candidate in self.population]
            next_generation = []

            for i in range(0, len(self.population), 2):
                parents_idx = self.selection_function()
                parents = [self.population[parents_idx[0]], self.population[parents_idx[1]]]
                children = self.crossover_function(parents[0], parents[1])

                for j in range(len(children)):
                    children[j] = self.mutation_function(children[j])
                    next_generation.append(children[j])

            self.population = next_generation
            strongest_candidate = self.get_best_individual(self.population) # in the current generation
            percentage_zero = (len(fitness_scores)-np.count_nonzero(np.array(fitness_scores)))/len(fitness_scores)
            self.scores.append((iteration,
                                min(fitness_scores),
                                sum(fitness_scores)/len(fitness_scores),
                                max(fitness_scores),
                                percentage_zero,
                                strongest_candidate))
            if self. maximization:
                self.best_score = max(max(fitness_scores), self.best_score)
            else:
                self.best_score = min(min(fitness_scores), self.best_score)
            self.best_individual = self.get_best_individual(self.population + [self.best_individual])
            if iteration % print_delay == 0:
                print(f"Iteration {iteration}: "
                      f"Max Fitness: {round(max(fitness_scores),2)}, "
                      f"Avg Fitness: {round(sum(fitness_scores)/len(fitness_scores),2)}, "
                      f"Min Fitness: {round(min(fitness_scores),2)}, "
                      f"Best score so far: {round(self.best_score, 2)}")

        return self.get_best_individual(self.population)

    def selection_function(self):
        """
        Performs selection process on the population and returns the indices of the individuals
        which are chosen to be parents for the next generation
        """

        fitness_scores = [self.fitness_function(candidate) for candidate in self.population]
        if self.selection_type == 'proportional':
            return self.proportional_selection(fitness_scores)
        elif self.selection_type == 'tournament':
            return self.tournament_selection()
        elif self.selection_type == 'exp_rank':
            return self.exp_rank_selection()
        elif self.selection_type == 'rank':
            return self.rank_selection()

    def tournament_selection(self, tournament_size=5):
        selected_parents = []
        parents_pool = [x for x in self.population]
        population_size = len(parents_pool)

        for _ in range(2):
            tournament_candidates = random.sample(parents_pool, min(tournament_size, population_size))
            selected_parent = self.get_best_individual(tournament_candidates)
            selected_parents.append(selected_parent)
            parents_pool.remove(selected_parent)

        parents_ids = [self.population.index(selected_parent) for selected_parent in selected_parents]

        return parents_ids

    def proportional_selection(self, fitness_scores):
        weights = []
        total_score = np.sum(fitness_scores)
        if total_score == 0:
            return random.choices(range(len(fitness_scores)), k=2)
        for score in fitness_scores:
            weights.append(score / total_score)
        if not self.maximization:
            weights = [1-w for w in weights]
        return random.choices(range(len(fitness_scores)), weights=weights, k=2)

    def sort_population(self):
        sorted_population = sorted(self.population, key=self.fitness_function, reverse=self.maximization)
        return sorted_population

    def exp_rank_selection(self, c = 0.1, k = 0.1):
        """
        Select 2 parents using an exponential rank selection
        :param c: constant that determines the scale of selection probabilities
        :param k: constant controlling the shape of the probability distribution
        :return:
        """
        sorted_population = self.sort_population()
        ranks = np.arange(1, len(sorted_population)+1)
        probs = c * (1 - np.exp(-ranks/k))
        selected_parents = random.choices(sorted_population, weights=probs, k=2)
        parents_ids = [self.population.index(selected_parent) for selected_parent in selected_parents]
        return parents_ids

    def rank_selection(self):
        sorted_population = self.sort_population()
        total_rank = sum(range(1, len(self.population) + 1))
        population_size = (len(self.population))
        selection_probs = [(population_size - i) / total_rank for i in range(population_size)]
        selected_parents = random.choices(sorted_population, weights=selection_probs, k=2)
        parents_ids = [self.population.index(selected_parent) for selected_parent in selected_parents]
        return parents_ids


    def get_best_individual(self, candidates):
        if candidates is None:
            candidates = self.population
        if self.maximization:
            best_individual = max(candidates, key=self.fitness_function)
        else:
            best_individual = min(candidates, key=self.fitness_function)

        return best_individual

    def crossover_function(self, parent1, parent2, type='one_point'):
        if not self.random_check(self.crossover_probability):
            return parent1, parent2

        if not self.custom_crossover is None:
            return self.custom_crossover(parent1, parent2)

        if type == 'one_point':
            return self.one_point_crossover(parent1, parent2)

    def one_point_crossover(self, parent1, parent2):
        split = random.randrange(start=0, stop=len(parent1) - 1)
        children = [parent1[:split] + parent2[split:], parent2[:split] + parent1[split:]]
        return children

    def mutation_function(self, child):
        if not self.random_check(self.mutation_probability):
            return child
        num_mutations = int(len(child)/10)
        for _ in range(num_mutations):
            if not self.custom_mutation is None:
                child = self.custom_mutation(child)
            else:
                child = self.classic_mutation(child)
        return child

    @staticmethod
    def classic_mutation(child):
        bit = random.randrange(start=0, stop=len(child) - 1)
        if child[bit] == 1:
            child[bit] = 0
        else:
            child[bit] = 1
        return child

    @staticmethod
    def random_check(threshold):
        if threshold == 1:
            return True

        random_var = random.uniform(0,1)
        if random_var <= threshold:
            return True
        return False

