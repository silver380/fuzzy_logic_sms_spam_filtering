import util
import numpy as np
import time
from evolutionary_algorithm import EvolutionaryAlgorithm as EA
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, matthews_corrcoef
import seaborn as sb
import matplotlib.pyplot as plt

N_ITER = util.n_iter
POPULATION_SIZE = 10
MUT_PROB = 0.9
RECOMB_PROB = 0.9
MAX_RULES = 50

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

   y_hat = ans.test(X_test)
   y_hat = np.array(y_hat)
   cm_train = confusion_matrix(y_test, y_hat)
   plt.subplots(figsize=(10, 6))
   sb.heatmap(cm_train, annot = True, fmt = 'g')
   plt.xlabel("Predicted")
   plt.ylabel("Actual")
   plt.title("Confusion Matrix for the training set")
   plt.show()
   print(accuracy_score(y_test,y_hat))
   print(f"matthews_corrcoef: {matthews_corrcoef(y_test, y_hat)}")

   #print(fitness_history)