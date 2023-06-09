import math
import numpy as np

curr_iter = 0
n_iter = 100

def isosceles_triangular(x, s, m):
    numerator1 = x - m + s
    numerator2 = m - x + s
    denominator = s
    return max(0, min(numerator1 / denominator, numerator2 / denominator))

def right_angled_trapezoidal(x, s, m):
    numerator1 = x - m + s
    denominator = s
    return max(0, min(numerator1 / denominator, 1))

def gaussian(x, s, m):
    exponent = -0.5 * ((x - m) / s) ** 2
    return math.exp(exponent + 1e-11)

def sigmoid(x, s, m):
    exponent = ((x - m) / s)
    if exponent < 0 :
        ans = np.exp(exponent)/(1+math.exp(exponent))
    else:
        ans = 1 / (1 + math.exp(-exponent))
    return ans

def calculate_k(population_size, iter):
    return max(2, population_size * iter // n_iter)