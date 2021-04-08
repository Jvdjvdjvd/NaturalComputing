#!/usr/bin/env python3

from math import factorial, floor

def probability_mass_function(x, c, p):
  a = factorial(c) / (factorial(x) * factorial(c - x))
  b = p**x * (1 - p)**(c - x)
  return a * b

def majority_wins(c, p):
  return sum([probability_mass_function(k, c, p) for k in range(floor(c / 2) + 1, c + 1)])

if __name__ == '__main__':
  p = 0.6
  c = 31

  print(majority_wins(c, p))
