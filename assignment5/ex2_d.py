#!/usr/bin/env python3
from ex2_b import probability_mass_function, majority_wins

if __name__ == '__main__':
  p_radiologist = 0.85
  c_radiologist = 1
  res_radiologist = majority_wins(c_radiologist, p_radiologist)
  print(f"probability correct {c_radiologist} radiologist: {res_radiologist}")

  p_doctors = 0.75
  c_doctors = 3
  res_doctors = majority_wins(c_doctors, p_doctors)
  print(f"probability correct {c_doctors} doctors: {res_doctors}")

  p_students = 0.6
  c_students = 31
  res_students = majority_wins(c_students, p_students)
  print(f"probability correct {c_students} students: {res_students}")

  # Find amount of students that result in their success probability being
  # as close as possible to the 3 doctors
  smallest_delta = 1000
  c_students_closest_to_doctors = -1
  res_students_closest_to_doctors = -1
  for c in range(100):
    res_students = majority_wins(c, p_students)
    delta = abs(res_students - res_doctors)
    print(res_students)
    if delta < smallest_delta:
      c_students_closest_to_doctors = c
      res_students_closest_to_doctors = res_students
      smallest_delta = delta

  print(f"The number of students needs to be {c_students_closest_to_doctors} for their probability to be as close as possible to the doctors.")
  print(f"the students then have a succes probability of {res_students_closest_to_doctors}, with delta {smallest_delta} to the 3 doctors")
