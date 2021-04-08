#!/usr/bin/env python3
import numpy as np
from ex2_b import probability_mass_function, majority_wins
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__ == '__main__':
  STOP = 100

  p_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  c_vector = list(range(1,STOP))
  xi = list(range(1,STOP))
  results = np.zeros((len(c_vector), len(p_vector)))

  for i, p in enumerate(p_vector):
    for j, c in enumerate(c_vector):
      results[j,i] = majority_wins(c, p)

  plt.plot(results)
  plt.xlabel("jury size c")
  plt.ylabel("probability majority is correct")
  plt.xticks(np.arange(0, len(c_vector)+1, 5))
  plt.legend(p_vector)
  plt.title("probability majority is correct for variable competence p and jury size c")
  plt.savefig("confidence_and_jury.png")
