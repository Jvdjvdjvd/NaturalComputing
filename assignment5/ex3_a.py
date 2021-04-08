#!/usr/bin/env python3
from ex2_b import probability_mass_function

def least_correct(least_correct, c, p):
  return sum([probability_mass_function(k, c, p) for k in range(least_correct, c + 1)])

if __name__ == '__main__':
  correct_5 = least_correct(5, 10, 0.6)
  correct_6 = least_correct(6, 10, 0.6)
  print(f"probability where at least 5 weak are correct {correct_5}")
  print(f"probability where at least 6 weak are correct {correct_6}")
