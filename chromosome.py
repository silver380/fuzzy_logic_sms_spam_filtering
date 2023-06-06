import random
import util
import math


class Chromosome:
    def __init__(self, mut_prob, recomb_prob, max_rules, calc_fitness):
        self.ferules = {"f1": [], "f2": [], "f3": [], "f4": [], "f5": [], "rule_base":[]}
        # Mutation probability
        self.mut_prob = mut_prob

        # Recombination probability
        self.recomb_prob = recomb_prob

        self.max_rules = max_rules

        # The maximum bandwidth of the towers
        self.fitness = 0
        self.calc_fitness = calc_fitness
        self.init_chromosome()

    def generate_s_m_MF(self):
        s = random.uniform(-4, 4)
        m = random.uniform(-2, 2)
        MF = random.randint(1, 4)
        return s, m, MF
    
    def init_chromosome(self):
        for i in range(5):
            num_ling_var = random.randint(3, 5)
            for _ in range(num_ling_var):
                s, m, MF = self.generate_s_m_MF()
                self.ferules[f'f{i+1}'].append((s, m, MF))

        for _ in range(self.max_rules):
            rule = []
            for i in range(5):
                neg = -1 if random.uniform(0, 1) <= 0.5 else 1
                rule.append(neg * random.randint(0, len(self.ferules[f'f{i+1}'])))
            rule.append(random.randint(0, 1))
            self.ferules['rule_base'].append(rule)
        
        if self.calc_fitness:
            self.calculate_fitness()

    def mut_rule_append(self):
        rule_append_prob = random.uniform(0, 1)
        if rule_append_prob <= self.mut_prob:
            rule = []
            for i in range(5):
                neg = -1 if random.uniform(0, 1) <= 0.5 else 1
                rule.append(neg * random.randint(0, len(self.ferules[f'f{i+1}'])))
            rule.append(random.randint(0, 1))
            self.ferules['rule_base'].append(rule)

    def mut_rule_pop(self):
        rule_pop_prob = random.uniform(0, 1)
        if rule_pop_prob <= self.mut_prob:
            pop_id = random.randint(0, len(self.ferules['rule_base']) - 1)
            self.ferules['rule_base'].pop(pop_id)

    def mut_rule_change(self):
        for k in range(len(self.ferules['rule_base'])):
            rule_change_prob = random.uniform(0, 1)
            if rule_change_prob <= self.mut_prob:
                i = random.randint(0, 5)
                if i != 5:
                    neg = -1 if random.uniform(0, 1) <= 0.5 else 1
                    self.ferules['rule_base'][k][i] = neg * random.randint(0, len(self.ferules[f'f{i+1}']))
                else:
                    self.ferules['rule_base'][k][i] = random.randint(0, 1)

    def mut_feature_append(self):
        for i in range(5):
            if len(self.ferules[f'f{i+1}']) < 5:
                feature_append_prob = random.uniform(0, 1)
                if feature_append_prob <= self.mut_prob:
                    s, m, MF = self.generate_s_m_MF()
                    self.ferules[f'f{i+1}'].append((s, m, MF))

    def mut_feature_pop(self):
        for i in range(5):
            if len(self.ferules[f'f{i+1}']) > 3:
                feature_pop_prob = random.uniform(0, 1)
                if feature_pop_prob <= self.mut_prob:
                    pop_id = random.randint(0, len(self.ferules[f'f{i+1}']) - 1)
                    self.ferules[f'f{i+1}'].pop(pop_id)

    def mut_feature_change(self):
        for i in range(5):
            feature_change_prob = random.uniform(0, 1)
            if feature_change_prob <= self.mut_prob:
                idx = random.randint(0, len(self.ferules[f'f{i+1}']) - 1)
                s, m, MF = self.generate_s_m_MF()
                self.ferules[f'f{i+1}'][idx] = (s, m, MF)

    def mutation(self):
        self.mut_feature_pop()
        self.mut_feature_change()
        self.mut_feature_append()

        self.mut_rule_pop()
        self.mut_rule_change()
        self.mut_rule_append()
        self.calculate_fitness()

    def calculate_fitness(self):
        self.fitness = random.randint(0, 10)        