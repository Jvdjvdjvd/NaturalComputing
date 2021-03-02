#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import tempfile as tmp
import os as os
from subprocess import Popen, PIPE, STDOUT
from time import sleep

def substringer(windows_size, data: str) -> list : # parses string and return the possible substrings
    substrings = [data[i:i+windows_size] for i in range(len(data)-windows_size+1)]
    return substrings

def sliding_window(windows_size, data: list) -> list: # gives the substrings of a lsit of strings (not used)
    substrings = []
    for d in data:
        substrings += [substringer(windows_size, d)]
    return substrings

def chunk(chunk_size, data: list) -> list: # separate the strings in a list to chunks
    substrings = []
    for d in data:
        local_substrings = []
        for i in range(0, len(d), chunk_size):
            chunk = d[i:(i + chunk_size)]
            if len(chunk) != chunk_size:
                chunk = d[-chunk_size:]
            local_substrings.append(chunk)
        substrings.append(local_substrings)

    return substrings


def run_negative_selection(training_file, n, r, testing_data, alphabet_file = None): # performs the negative selection
    cmd = ['java', '-jar', './negative-selection/negsel2.jar', '-self', training_file, '-n', str(n), '-r', str(r), '-c', '-l']
    if alphabet_file:
        cmd = cmd + ['-alphabet', f'file://{alphabet_file}']

    in_data = bytes('\n'.join(testing_data), encoding="raw_unicode_escape") + b'\x1a'
    p = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    p.stdin.write(in_data)
    res = p.communicate()[0].decode()

    return [float(s) for s in res.splitlines()]

def get_AUC_from_anomalies(data: list, savename = 'temp.png'): 
    """
    :param data: list of tuples (label, word, score)
    :param savename: name to save to plot as
    :return: AUC
    """

    sorted_data = sorted(data, key=lambda x: x[2])
    l = len(sorted_data)

    # create list of distinct scores for higher/lower comparing
    distinct_scores = []
    prev_i = 0
    prev_score = sorted_data[0][2]
    for i, d in enumerate(sorted_data):
        (_, _, score) = d
        if prev_score != score:
            distinct_scores.append((prev_i, i))
            prev_i = i
            prev_score = score
    distinct_scores.append((prev_i, l))

    # get amount of normal / abnormal scores
    l_anom = len([w for (l, w, s) in sorted_data if not l])
    l_norm = len([w for (l, w, s) in sorted_data if l])

    sensitivities = []
    specificities = []
    for i, j in distinct_scores:
        score = sorted_data[i][2]
        sensitivity = len([w for (l, w, s) in sorted_data if s >= score and not l])
        specificity = len([w for (l, w, s) in sorted_data if s < score and l])

        sensitivities.append(sensitivity / l_anom)
        specificities.append(specificity / l_norm)

    # calculate AUC and make plots
    inverse_spec = np.ones_like(specificities) - specificities
    auc = metrics.auc(inverse_spec, sensitivities)

    plt.figure()
    plt.plot(inverse_spec, sensitivities)
    plt.plot([0.0,1.0], [0.0,1.0], 'r--')
    plt.xlabel('False positive rate (1-specificity)')
    plt.ylabel('True positive rate (Sensitivity)')
    plt.title(f'ROC curve (AUC = {auc})')
    plt.savefig(savename)
    plt.close()

    return auc

def run_for_training_set(training_file, alphabet_file, test_data_files):
    testing_words = []
    testing_labels = []
    testing_scores = []

    # determine the smallest length of the 'words'
    max_n = None
    for d in [d + ".test" for d in test_data_files] + [training_file]:
        with open(d, 'r') as f:
            x = min([len(l.strip()) for l in f.readlines()])
            if max_n is None or max_n > x:
                max_n = x

    print(f"Maximum n: {max_n}")

    # loop though multiple r and n values
    for n in range(3, max_n):
        for r in range (3, n):
            processed_train_file = "" # transform train file to chucks   
            with open(training_file, 'r') as f:
                train_lines = [x.strip() for x in f.readlines()]
                train_data = chunk(n, train_lines)
                fd, path = tmp.mkstemp()
                print(f"using tmp file {path}")
                with os.fdopen(fd, 'w') as t:
                    t.write("\n".join(["\n".join(j) for j in train_data]))
                processed_train_file = path

            for d in test_data_files: # read the test files and test labels
                name = d.split("/")[-1]
                print(f"testing file: {d}")
                labels = None
                with open(d + ".labels", 'r') as f:
                    labels = [not(bool(int(x.strip()))) for x in f.readlines()]
                    testing_labels.append(labels)

                local_testing_words = []
                local_testing_scores = []
                with open(d + ".test", 'r') as f:
                    test_lines = [x.strip() for x in f.readlines()]
                    test_data = chunk(n, test_lines) # create chunks from the test files

                    flat_test_data = [e for l in test_data for e in l] # perform negative selection on the test chunks
                    scores = run_negative_selection(processed_train_file, n, r, flat_test_data, alphabet_file=alphabet_file)
                    j = 0
                    for td in test_data: # average the scores
                        test_data_score = np.average(scores[j : (j + len(td))])
                        local_testing_scores.append(test_data_score)
                        local_testing_words.append(td)
                        j = j + len(td)

                    testing_words.append(local_testing_words)
                    testing_scores.append(local_testing_scores)

                auc_data = list(zip(labels, local_testing_words, local_testing_scores)) # determine the AUC

                auc = get_AUC_from_anomalies(auc_data, f"{name}-{n}-{r}.png")
                print(f"{name}, auc: {auc}")

if __name__ == '__main__':
    training_file = "./negative-selection/syscalls/snd-unm/snd-unm.train"
    alphabet_file = "./negative-selection/syscalls/snd-unm/snd-unm.alpha"
    
    test_data_files = [
        "./negative-selection/syscalls/snd-unm/snd-unm.1",
        "./negative-selection/syscalls/snd-unm/snd-unm.2",
        "./negative-selection/syscalls/snd-unm/snd-unm.3"
    ]

    run_for_training_set(training_file, alphabet_file, test_data_files)

    training_file = "./negative-selection/syscalls/snd-cert/snd-cert.train"
    alphabet_file = "./negative-selection/syscalls/snd-cert/snd-cert.alpha"
    
    test_data_files = [
        "./negative-selection/syscalls/snd-cert/snd-cert.1",
        "./negative-selection/syscalls/snd-cert/snd-cert.2",
        "./negative-selection/syscalls/snd-cert/snd-cert.3"
    ]

    run_for_training_set(training_file, alphabet_file, test_data_files)
