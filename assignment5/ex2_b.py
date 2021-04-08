#!/usr/bin/env python3

from math import factorial

def probability_mass_function(k, n, p):
  a = factorial(n) / (factorial(k) * factorial(n - k))
  b = p**k * (1 - p)**(n - k)
  return a * b

p = 0.6
n = 31

majority_wins = sum([probability_mass_function(k, n, p) for k in range(int(n / 2) + 1, n + 1)])
print(majority_wins)
