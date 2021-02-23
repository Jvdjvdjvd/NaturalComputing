#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE, STDOUT
from signal import SIGINT
from time import sleep


def run_negative_selection(training_file, testing_set):
    cmd = ['java', '-jar', './negative-selection/negsel2.jar', '-self', training_file, '-n', '10', '-r', '4', '-c', '-l']
    in_data = bytes('\n'.join(testing_set), encoding="raw_unicode_escape") + b'\x1a'
    p = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    p.stdin.write(in_data)
    result = p.communicate()[0]
    return [float(s) for s in result.decode().splitlines()]

def get_AUC_from_anomalities(anomalies):
    sorted_anomalies = np.sort(anomalies)
    
    l = len(sorted_anomalies)

    distinct_anomalies = []
    prev_i = 0
    prev_ele = sorted_anomalies[0]
    for i, cut_off in enumerate(sorted_anomalies):
        if prev_ele != cut_off:
            distinct_anomalies.append((prev_i, i, prev_ele))
            prev_i = i
            prev_ele = cut_off

    print(distinct_anomalies)
    print("---------")

    sensitivity = []
    specificity = []
    for i, j, cut_off in distinct_anomalies:
        sens = 100*(l - j)/l #sensitivity = how many higher values
        spec = 100*(i/l) #amount with lower score
        sensitivity.append(sens)
        specificity.append(spec)

    print(sensitivity)
    print("---------")
    print(specificity)

    inverse_spec = 100*np.ones_like(specificity) - specificity
    plt.plot(inverse_spec, sensitivity)
    plt.plot([0,100], [0,100], 'r--')
    plt.xlabel('False positive rate (1-specificity)')
    plt.ylabel('True positive rate (Sensitivity)')
    plt.title(
        'AUC curve')
    plt.savefig('temp.png')

    return None

if __name__ == '__main__':
    training_file = "./negative-selection/english.train"

    testing_english = None
    testing_tagalog = None

    with open('negative-selection/tagalog.test', 'r') as f:
        testing_tagalog = [s.strip() for s in f.readlines()]
    
    with open('negative-selection/english.test', 'r') as f:
        testing_english = [s.strip() for s in f.readlines()]

    total_set = testing_tagalog + testing_english

    anomilies = run_negative_selection(training_file, total_set)

    auc = get_AUC_from_anomalities(anomilies)
