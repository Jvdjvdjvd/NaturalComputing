#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from math import log

if __name__ == '__main__':
  errs = np.linspace(0.01, 0.99, num = 99)
  results = np.zeros(len(errs))

  for i, err in enumerate(errs):
    results[i] = log((1.0 - err)/err, 10)

  plt.plot(errs, results)
  plt.xlabel("Error")
  plt.ylabel("Weight")
  plt.title("AdaBoost Error vs Weight")
  plt.savefig("AdaBoost_error_vs_weight.png")
