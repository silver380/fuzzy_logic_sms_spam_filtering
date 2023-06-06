import util
import numpy as np
import time
from evolutionary_algorithm import EvolutionaryAlgorithm as EA


N_ITER = 20
POPULATION_SIZE = 50
MUT_PROB = 0.9
RECOMB_PROB = 0.1
MAX_RULES = 10

if __name__ == "__main__":
   ea = EA(N_ITER, MUT_PROB, RECOMB_PROB, POPULATION_SIZE, MAX_RULES)
   ans, fitness, fitness_history = ea.run()
   print(fitness_history)