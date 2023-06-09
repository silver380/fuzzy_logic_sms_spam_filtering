import util
import numpy as np
import random
from chromosome import Chromosome
import sys
import copy

class EvolutionaryAlgorithm:
    def __init__(self, n_iter, mut_prob, recomb_prob, population_size, max_rules, data):
        self.n_iter = n_iter
        self.mut_prob = mut_prob
        self.recomb_prob = recomb_prob
        self.max_rules = max_rules
        self.population_size = population_size
        self.population = []
        self.data = data
        self.current_iter = 0
        self.fitness_avg = 0
        self.fitness_history = []

    # Random initialization
    def init_population(self):
        for _ in range(self.population_size):
            young_pop = Chromosome(self.mut_prob, self.recomb_prob, self.max_rules, True, data=self.data)
            self.population.append(young_pop)

    # Fitness Tournament selection
    def tournament_selection(self, tour_pop, k):
        parents = random.sample(tour_pop, k=k)
        parents = sorted(parents, key=lambda agent: agent.fitness, reverse=True)
        bestparent = parents[0]
        return bestparent

    def parent_selection(self):
        parents = []
        for _ in range(self.population_size):
            best_parent = self.tournament_selection(self.population,
                                                    util.calculate_k(len(self.population), self.current_iter))
            parents.append(best_parent)

        return parents

    def error_correction(self, young):
        for k in range(len(young.ferules['rule_base'])):
            for i in range(5):
                if abs(young.ferules['rule_base'][k][i]) > len(young.ferules[f'f{i}']):
                    neg = -1 if random.uniform(0, 1) <= 0.5 else 1
                    young.ferules['rule_base'][k][i] = neg * (random.randint(0, len(young.ferules[f"f{i}"])))
        return young
        
    def rule_base_recombination(self, parent_1, parent_2, young1, young2):
        crossover_point = random.randint(1, self.max_rules - 2)
        young1.ferules['rule_base'] = copy.deepcopy(parent_1.ferules['rule_base'][:crossover_point]) + copy.deepcopy(parent_2.ferules['rule_base'][crossover_point:])
        young2.ferules['rule_base'] = copy.deepcopy(parent_2.ferules['rule_base'][:crossover_point]) + copy.deepcopy(parent_1.ferules['rule_base'][crossover_point:])

        return young1, young2

    def feature_recombination(self, parent_1, parent_2, young1, young2):
        crossover_point = random.randint(0, 4)
        for i in range(crossover_point):
            young1.ferules[f"f{i}"] = parent_1.ferules[f"f{i}"].copy()
            young2.ferules[f"f{i}"] = parent_2.ferules[f"f{i}"].copy()

        for i in range(crossover_point, 5):
            young1.ferules[f"f{i}"] = parent_2.ferules[f"f{i}"].copy()
            young2.ferules[f"f{i}"] = parent_1.ferules[f"f{i}"].copy()

        return young1, young2
    

    def recombination(self, mating_pool):
        youngs = []
        for _ in range(self.population_size // 2):
            parents = random.choices(mating_pool, k=2).copy()
            young1 = Chromosome(self.mut_prob, self.recomb_prob, self.max_rules, False, self.data)
            young2 = Chromosome(self.mut_prob, self.recomb_prob, self.max_rules, False, self.data)

            prob = random.uniform(0, 1)
            if prob <= self.recomb_prob:
                young1, young2 = self.feature_recombination(parents[0], parents[1], young1, young2)
                young1, young2 = self.rule_base_recombination(parents[0], parents[1], young1, young2)

            else:
                young1.ferules = parents[0].ferules.copy()
                young2.ferules = parents[1].ferules.copy()


            youngs.append(young1)
            youngs.append(young2)

        return youngs

    def all_mutation(self, youngs):
        for i in range(len(youngs)):
            youngs[i].mutation()
            
        
        return youngs

    def survival_selection(self, youngs):
        mpl = self.population.copy() + youngs
        mpl = sorted(mpl, key=lambda agent: agent.fitness, reverse=True)
        mpl = mpl[:self.population_size].copy()
        return mpl

    def calculate_fitness_avg(self):
        self.fitness_avg = 0
        for pop in self.population:
            self.fitness_avg += pop.fitness

    def run(self):
        self.init_population()
        prev_avg = 0

        for _ in range(self.n_iter):
            parents = self.parent_selection().copy()
            youngs = self.recombination(parents).copy()
            youngs = self.all_mutation(youngs).copy()
            self.population = self.survival_selection(youngs).copy()
            self.calculate_fitness_avg()
            self.current_iter += 1
            util.curr_iter += 1
            best_current = sorted(self.population, key=lambda agent: agent.fitness, reverse=True)[0]
            print(f"current iteration: {self.current_iter} / {self.n_iter}", f", best fitness: {best_current.fitness}")
            print(f'fitness_avg: {self.fitness_avg / (self.population_size)}')
            print("--------------------------------------------------------------------------------------------------")
            self.fitness_history.append(self.fitness_avg / (self.population_size))
            prev_avg = self.fitness_avg / (self.population_size)

        ans = sorted(self.population, key=lambda agent: agent.fitness, reverse=True)[0]

        return ans, ans.fitness, self.fitness_history