import random
import util
import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, matthews_corrcoef
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
        self.label_cnt = [0,0]
        self.max_label_diff = max_rules / 10
        self.calc_fitness = calc_fitness
        self.init_chromosome()

    def generate_s_m_mf(self):
        m = random.uniform(1, 12)
        s = random.uniform(0, m/2)
        mf = random.randint(1, 4)
        return s, m, mf
    
    def init_chromosome(self):
        for i in range(5):
            num_ling_var = random.randint(3, 5)
            for _ in range(num_ling_var):
                s, m, mf = self.generate_s_m_mf()
                self.ferules[f"f{i}"].append((s, m, mf))

        for _ in range(self.max_rules):
            rule = []
            for i in range(5):
                neg = -1 if (random.uniform(0, 1) <= 0.5) else 1
                rule.append((neg * random.randint(0, len(self.ferules[f'f{i}']))))

            y = random.randint(0, 1)
            if abs(self.label_cnt[y] - (self.label_cnt[1-y])) > self.max_label_diff:    
                y = 1-y if (self.label_cnt[y] > (self.label_cnt[1-y])) else y

            self.label_cnt[y] += 1
            rule.append(y)
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
                if i < 5:
                    neg = -1 if (random.uniform(0, 1) <= 0.5) else 1
                    self.ferules['rule_base'][k][i] = (neg * (random.randint(0, len(self.ferules[f"f{i}"]))))
                else:
                    y = random.randint(0, 1)
                    if self.ferules['rule_base'][k][i] == y:
                        continue

                    if abs(self.label_cnt[y] + 1 - (self.label_cnt[1-y] - 1)) > self.max_label_diff:
                        y = 1-y if (self.label_cnt[y] + 1 > (self.label_cnt[1-y] - 1)) else y
                    else:
                        self.label_cnt[y] += 1
                        self.label_cnt[1-y] -= 1

                    self.ferules['rule_base'][k][i] = y

    def mut_feature_append(self):
        for i in range(5):
            if len(self.ferules[f"f{i}"]) < 5:
                feature_append_prob = random.uniform(0, 1)
                if feature_append_prob <= self.mut_prob:
                    s, m, mf = self.generate_s_m_mf()
                    self.ferules[f"f{i}"].append((s, m, mf))

    def mut_feature_pop(self):
        for i in range(5):
            if len(self.ferules[f"f{i}"]) > 3:
                feature_pop_prob = random.uniform(0, 1)
                if feature_pop_prob <= self.mut_prob:
                    pop_id = random.randint(0, len(self.ferules[f"f{i}"]) - 1)
                    self.ferules[f"f{i}"].pop(pop_id)

    def mut_feature_change(self):
        for i in range(5):
            feature_change_prob = random.uniform(0, 1)
            if feature_change_prob <= self.mut_prob:
                idx = random.randint(0, len(self.ferules[f"f{i}"]) - 1)
                s, m, mf = self.generate_s_m_mf()
                self.ferules[f"f{i}"][idx] = (s, m, mf)
    
    def error_correction(self):
        for k in range(len(self.ferules['rule_base'])):
            for i in range(5):
                if abs(self.ferules['rule_base'][k][i]) > len(self.ferules[f"f{i}"]):
                    neg = -1 if (random.uniform(0, 1) <= 0.5) else 1
                    self.ferules['rule_base'][k][i] = neg * (random.randint(0, len(self.ferules[f"f{i}"])))

    def mutation(self):
        self.label_cnt[0] = 0
        self.label_cnt[1] = 0
        for rule in self.ferules['rule_base']:
            self.label_cnt[rule[5]] += 1

        self.mut_feature_pop()
        self.mut_feature_change()
        self.mut_feature_append()

        #self.mut_rule_pop()
        
        self.mut_rule_change()
        #self.mut_rule_append()
        
        self.error_correction()
        
        self.calculate_fitness()

    def membership(self, x, f, negated):

        s,m,mf = f[0],f[1],f[2]
        ans = 0
        if mf == 1: # Isosceles Triangular
            numerator1 = x - m + s
            numerator2 = m - x + s
            denominator = s
            ans = max(0, min(numerator1 / denominator, numerator2 / denominator, 0))
        
        elif mf == 2: # Right-angled Trapezoidal
            numerator1 = x - m + s
            denominator = s
            ans = max(0, min(numerator1 / denominator, 1))

        elif mf == 3: # Gaussian
            exponent = -0.5 * ((x - m) / s) ** 2
            ans = math.exp(exponent + 1e-11)
        
        elif mf == 4: # Sigmoid
            exponent = ((x - m) / s)
            if exponent < 0 :
                ans = np.exp(exponent)/(1+math.exp(exponent))
            else:
                ans = 1 / (1 + math.exp(-exponent))
        
        if negated:
            ans = 1 - ans
        
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
                        negated = False if (rule[a] >= 0) else True
                        mu.append(self.membership(x[a],self.ferules[f"f{a}"][abs(rule[a])-1],negated))
                
                gr = self.agg_algebric_product(mu)
                if(rule[5]==0):
                    gc_x[0] += gr
                else:
                    gc_x[1] += gr
            
            y_hat.append(np.argmax(gc_x))

        y_hat = np.array(y_hat.copy())
        #self.fitness = accuracy_score(self.data[1],y_hat)
        self.fitness = matthews_corrcoef(self.data[1], y_hat)
        
    def test(self, X_test):
        y_hat = []
        for x in X_test:
            gc_x = [0,0]
            
            for rule in self.ferules['rule_base']:
                
                gr = 0
                mu = []
                for a in range(5):
                    if rule[a] != 0:
                        negated = False if rule[a] >= 0 else True
                        try:
                            mu.append(self.membership(x[a],self.ferules[f"f{a}"][abs(rule[a])-1],negated))
                        except:
                            print(f"bad index: {a}, {rule}, {rule[a]}")
                
                gr = self.agg_algebric_product(mu)
                if(rule[5]==0):
                    gc_x[0] += gr
                else:
                    gc_x[1] += gr
            
            y_hat.append(np.argmax(gc_x))

        y_hat = np.array(y_hat.copy())
        return y_hat
    
