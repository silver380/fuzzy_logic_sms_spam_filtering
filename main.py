import util
import numpy as np
import time
from evolutionary_algorithm import EvolutionaryAlgorithm as EA


N_ITER = 10
POPULATION_SIZE = 50
MUT_PROB = 0.9
RECOMB_PROB = 0.6
MAX_RULES = 100

if __name__ == "__main__":
   X_train = np.load('X_train.npy')
   X_test  = np.load('X_test.npy')
   y_train = np.load('y_train.npy')
   y_test = np.load('y_test.npy')
   data = [X_train, y_train]
   ea = EA(N_ITER, MUT_PROB, RECOMB_PROB, POPULATION_SIZE, MAX_RULES, data)
   ans, fitness, fitness_history = ea.run()

   print(f"f0: {len(ans.ferules['f0'])}")
   print(f"f1: {len(ans.ferules['f1'])}")
   print(f"f2: {len(ans.ferules['f2'])}")
   print(f"f3: {len(ans.ferules['f3'])}")
   print(f"f4: {len(ans.ferules['f4'])}")
   for rule in ans.ferules['rule_base']:
      print(rule)

   print(f"test f1: {ans.test(X_test, y_test)}")
   #print(fitness_history)