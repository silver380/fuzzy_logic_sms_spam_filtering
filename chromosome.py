import random
import util
import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt
class Chromosome:
    def __init__(self, mut_prob, recomb_prob, max_rules, calc_fitness, data):
        self.ferules = {"f0": [], "f1": [], "f2": [], "f3": [], "f4": [], "rule_base":[]}
        # Mutation probability
        self.mut_prob = mut_prob

        # Recombination probability
        self.recomb_prob = recomb_prob

        self.max_rules = max_rules

        # X_train ad y_train
        self.data = data

        # The maximum bandwidth of the towers
        self.fitness = 0
        self.calc_fitness = calc_fitness
        self.init_chromosome()

    def generate_s_m_MF(self):
        s = random.uniform(-4, 4)
        m = random.uniform(-2, 2)
        MF = random.randint(1, 2)
        return s, m, MF
    
    def init_chromosome(self):
        for i in range(5):
            num_ling_var = random.randint(3, 5)
            for _ in range(num_ling_var):
                s, m, MF = self.generate_s_m_MF()
                self.ferules[f'f{i}'].append((s, m, MF))

        for _ in range(self.max_rules):
            rule = []
            for i in range(5):
                neg = -1 if random.uniform(0, 1) <= 0.5 else 1
                rule.append(neg * random.randint(0, len(self.ferules[f'f{i}'])))
            rule.append(random.randint(0, 1))
            self.ferules['rule_base'].append(rule.copy())
        
        if self.calc_fitness:
            self.calculate_fitness()

    def mut_rule_append(self):
        if len(self.ferules['rule_base']) < self.max_rules:
            rule_append_prob = random.uniform(0, 1)
            if rule_append_prob <= self.mut_prob:
                rule = []
                for i in range(5):
                    neg = -1 if random.uniform(0, 1) <= 0.5 else 1
                    rule.append(neg * random.randint(0, len(self.ferules[f'f{i}'])))
                rule.append(random.randint(0, 1))
                self.ferules['rule_base'].append(rule.copy())

    def mut_rule_pop(self):
        rule_pop_prob = random.uniform(0, 1)
        if len(self.ferules['rule_base']) > 1:
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
                    self.ferules['rule_base'][k][i] = neg * random.randint(0, len(self.ferules[f'f{i}']))
                else:
                    self.ferules['rule_base'][k][i] = random.randint(0, 1)

    def mut_feature_append(self):
        for i in range(5):
            if len(self.ferules[f'f{i}']) < 5:
                feature_append_prob = random.uniform(0, 1)
                if feature_append_prob <= self.mut_prob:
                    s, m, MF = self.generate_s_m_MF()
                    self.ferules[f'f{i}'].append((s, m, MF))

    def mut_feature_pop(self):
        for i in range(5):
            if len(self.ferules[f'f{i}']) > 3:
                feature_pop_prob = random.uniform(0, 1)
                if feature_pop_prob <= self.mut_prob:
                    pop_id = random.randint(0, len(self.ferules[f'f{i}']) - 1)
                    self.ferules[f'f{i}'].pop(pop_id)

    def mut_feature_change(self):
        for i in range(5):
            feature_change_prob = random.uniform(0, 1)
            if feature_change_prob <= self.mut_prob:
                idx = random.randint(0, len(self.ferules[f'f{i}']) - 1)
                s, m, MF = self.generate_s_m_MF()
                self.ferules[f'f{i}'][idx] = (s, m, MF)
    
    def error_correction(self):
        for k in range(len(self.ferules['rule_base'])):
            for i in range(5):
                if abs(self.ferules['rule_base'][k][i]) > len(self.ferules[f'f{i}']):
                    neg = -1 if random.uniform(0, 1) <= 0.5 else 1
                    self.ferules['rule_base'][k][i] = neg * random.randint(0, len(self.ferules[f'f{i}']))

    def mutation(self):
        self.mut_feature_pop()
        self.mut_feature_change()
        self.mut_feature_append()

        self.mut_rule_pop()
        self.mut_rule_change()
        self.mut_rule_append()
        
        self.error_correction()

        self.calculate_fitness()

    def membership(self, x, f, negated=False):

        s,m,MF = f[0],f[1],f[2]
        ans = 0
        if MF == 1: # Isosceles Triangular
            numerator1 = x - m + s
            numerator2 = m - x + s
            denominator = s
            ans = max(0, min(numerator1 / denominator, numerator2 / denominator, 0))
        
        elif MF == 2: # Right-angled Trapezoidal
            numerator1 = x - m + s
            denominator = s
            ans = max(0, min(numerator1 / denominator, 1))

        elif MF == 3: # Gaussian
            exponent = -0.5 * ((x - m) / s) ** 2
            ans = math.exp(exponent)
        
        elif MF == 4: # Sigmoid
            exponent = -((x - m) / s)
            ans = 1 / (1 + math.exp(exponent))
        
        if negated:
            return 1 - ans
        
        return ans

    def agg_algebric_product(self, mu):
        if len(mu) == 0:
            return 0
        return np.prod(mu)
    
    def agg_min(self, mu):
        if len(mu) == 0:
            return 0
        return np.min(mu)
    
    def calculate_fitness(self):   
        y_hat = []
        for x in self.data[0]:
            gc_x = [0,0]
            
            for rule in self.ferules['rule_base']:
                
                gr = 0
                mu = []
                for a in range(5):
                    if rule[a] != 0:
                        negated = False if rule[a] >= 0 else True
                        mu.append(self.membership(x[a],self.ferules[f'f{a}'][abs(rule[a])-1],negated))
                
                gr = self.agg_algebric_product(mu)
                if(rule[5]==0):
                    gc_x[0] += gr
                else:
                    gc_x[1] += gr
            
            y_hat.append(np.argmax(gc_x))

        y_hat = np.array(y_hat)
        #self.fitness = accuracy_score(self.data[1],y_hat)
        self.fitness = f1_score(self.data[1], y_hat, average='weighted')
        
    def test(self, X_test, y_test):
        y_hat = []
        for x in X_test:
            gc_x = [0,0]
            
            for rule in self.ferules['rule_base']:
                
                gr = 0
                mu = []
                for a in range(5):
                    if rule[a] != 0:
                        negated = False if rule[a] >= 0 else True
                        mu.append(self.membership(x[a],self.ferules[f'f{a}'][abs(rule[a])-1],negated))
                
                gr = self.agg_algebric_product(mu)
                if(rule[5]==0):
                    gc_x[0] += gr
                else:
                    gc_x[1] += gr
            
            y_hat.append(np.argmax(gc_x))

        y_hat = np.array(y_hat)
        cm_train = confusion_matrix(y_test, y_hat)
        plt.subplots(figsize=(10, 6))
        sb.heatmap(cm_train, annot = True, fmt = 'g')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix for the training set")
        plt.show()

        print(accuracy_score(y_test,y_hat))
        return f1_score(y_test, y_hat, average='weighted')

