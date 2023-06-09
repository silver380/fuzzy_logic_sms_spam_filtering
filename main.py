import util
import visualize
import numpy as np
import time
from evolutionary_algorithm import EvolutionaryAlgorithm as EA
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, matthews_corrcoef
import seaborn as sb
import matplotlib.pyplot as plt

N_ITER = util.n_iter
POPULATION_SIZE = 6
MUT_PROB = 0.9
RECOMB_PROB = 0.6
MAX_RULES = 50

if __name__ == "__main__":
   X_train = np.load('X_train.npy')
   X_test  = np.load('X_test.npy')
   y_train = np.load('y_train.npy')
   y_test = np.load('y_test.npy')
   data = [X_train, y_train]
   ea = EA(N_ITER, MUT_PROB, RECOMB_PROB, POPULATION_SIZE, MAX_RULES, data)
   ans, fitness, fitness_history = ea.run()

   for rule in ans.ferules['rule_base']:
      print(rule)
      
   for i in range(5):
      visualize.gen_membership_function(ans.ferules[f'f{i}'], i+1)
      
   visualize.gen_confusion_matrix(y_train, ans.train_y_hat, "training")
   visualize.gen_confusion_matrix(y_test, ans.test(X_test), "testing")