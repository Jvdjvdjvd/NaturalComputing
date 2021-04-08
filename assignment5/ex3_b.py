#!/usr/bin/env python3
from ex2_b import probability_mass_function
import numpy as np
import matplotlib.pyplot as plt
from math import floor

def least_correct(least_correct, c, p):
  return sum([probability_mass_function(k, c, p) for k in range(least_correct, c + 1)])

def evaluate_with_weight(w):
  strong_p = 0.75

  if w <= 0:
    w = 0.1

  if w > 10: # If we have higher weight then 10 we are majority voter
    w = 10.1

  count_pos = 5 - floor(w / 2)
  count_neg = 6 + floor(w / 2)
  if w % 2 == 0:
    count_pos = count_pos + 1

  a = strong_p * least_correct(count_pos, 10, 0.6)
  b = (1 - strong_p) * least_correct(count_neg, 10, 0.6)
  return a + b

if __name__ == '__main__':
  weights = np.linspace(0, 12, num = 121)
  results = np.zeros(len(weights))

  for i, w in enumerate(weights):
    results[i] = evaluate_with_weight(w)

  plt.plot(weights, results)
  plt.xlabel("Weight")
  plt.ylabel("probability weighted majority is correct")
  plt.title("probability weighted majority is correct for variable weights")
  plt.savefig("weighted_majority.png")
