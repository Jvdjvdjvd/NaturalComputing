#!/usr/bin/env python3

from math import factorial, ceil

def probability_mass_function(x, c, p):
  a = factorial(c) / (factorial(x) * factorial(c - x))
  b = p**x * (1 - p)**(c - x)
  return a * b

p = 0.6
c = 31

majority_wins = sum([probability_mass_function(x, c, p) for x in range(ceil(c / 2), c + 1)])
print(majority_wins)
