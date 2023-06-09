import util
import numpy as np
import time
from evolutionary_algorithm import EvolutionaryAlgorithm as EA
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, matthews_corrcoef
import seaborn as sb
import matplotlib.pyplot as plt
import visualize
N_ITER = util.n_iter
POPULATION_SIZE = 10
MUT_PROB = 0.9
RECOMB_PROB = 0.6
MAX_RULES = 50
histories = []
best_ans = None
repeat_num = 1

if __name__ == "__main__":
   X_train = np.load('X_train.npy')
   X_test  = np.load('X_test.npy')
   y_train = np.load('y_train.npy')
   y_test = np.load('y_test.npy')
   data = [X_train, y_train]
   i = 0 
   nq = 0
   while i < repeat_num:
      print(f"Run number {i+1} / {repeat_num}")
      ea = EA(N_ITER, MUT_PROB, RECOMB_PROB, POPULATION_SIZE, MAX_RULES, data)
      ans, fitness, fitness_history = ea.run()
      # t_fitness = matthews_corrcoef(y_test,ans.test(X_test))
      # if t_fitness < 0.25 and nq < 3:
      #    nq+=1
      #    continue
      # nq = 0
      if i > 0:
         if ans.fitness > best_ans.fitness:
            best_ans = ans
      else:
         best_ans = ans
         
      i+=1
      histories.append(fitness_history)


   histories = np.array(histories)   
   np.savetxt("histories.csv", histories,
               delimiter = ",")
   
   best_ans.explain(X_test[0],y_test[0])
   best_ans.print_rules()
   print(f"best found answer has {best_ans.fitness} fitness (MCC). on training Set")
   print(f"best found answer has {matthews_corrcoef(y_test,best_ans.test(X_test))} fitness (MCC). on test Set")
   for i in range(5):
      visualize.gen_membership_function(best_ans.ferules[f'f{i}'], i+1)
      
   visualize.gen_confusion_matrix(y_train, best_ans.train_y_hat, "training")
   visualize.gen_confusion_matrix(y_test, best_ans.test(X_test), "testing")

   #print(fitness_history)