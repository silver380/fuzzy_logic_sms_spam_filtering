import json
import numpy as np
import math

curr_iter = 0
n_iter = 200

def calculate_k(population_size, iter):
    return max(2, population_size * iter // n_iter)