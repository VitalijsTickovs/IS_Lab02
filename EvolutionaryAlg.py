import math
import random
import numpy as np


class EvolutionaryAlgorithm:

    def __init__(self, population, fitness_function):
        self.population = population
        self.fitness_function = fitness_function
        self.scores = []

    def evolve(self, generations):
        for iteration in range(generations):
            fitness_scores = [self.fitness_function(candidate) for candidate in self.population]
            children = []

            for i in range(0, len(self.population), 2):
                parents_idx = self.selection_function()
                parents = [self.population[parents_idx[0]], self.population[parents_idx[1]]]
                children = self.crossover_function(parents[0], parents[1])
                children.append(children[0])
                children.append(children[1])

                for j in range(len(children)):
                    children[j] = self.mutation_function(children[j])

            self.population = children
            self.scores.append((iteration, max(fitness_scores)))

    def selection_function(self, type='roulette'):
        fitness_scores = [self.fitness_function(candidate) for candidate in self.population]
        if type == 'roulette':
            return self.roulette_wheel(fitness_scores)
        elif type == 'tournament':
            return self.tournament_selection()

    def tournament_selection(self, tournament_size=5):
        selected_parents = []
        parents_pool = self.population
        population_size = len(parents_pool)

        for _ in range(2):
            tournament_candidates = random.sample(parents_pool, min(tournament_size, population_size))
            selected_parent = self.get_best_individual(tournament_candidates)
            selected_parents.append(selected_parent)
            selected_parent.remove(selected_parent)

        return selected_parents

    def roulette_wheel(self, fitness_scores):
        weights = []
        total_score = np.sum(fitness_scores)
        for score in fitness_scores:
            weights.append(score / total_score)
        return random.choices(range(len(fitness_scores)), weights=weights, k=2)

    def get_best_individual(self, candidates):
        if candidates is None:
            candidates = self.population
        best_individual = max(candidates, key=self.fitness_function)

        return best_individual

    def crossover_function(self, parent1, parent2, type='one_point'):
        if type == 'one_point':
            return self.one_point_crossover(parent1, parent2)

    def one_point_crossover(self, parent1, parent2):
        split = random.randrange(start=0, stop=len(parent1) - 1)
        children = [parent1[:split] + parent2[split:], parent2[:split] + parent1[split:]]
        return children

    def mutation_function(self, child):
        bit = random.randrange(start=0, stop=len(child) - 1)
        new_weight = random.random() * 10
        new_weight = math.floor(new_weight * 100) / 100
        child[bit] = new_weight
