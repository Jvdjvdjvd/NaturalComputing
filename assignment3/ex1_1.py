#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from subprocess import Popen, PIPE, STDOUT
from time import sleep

def run_negative_selection(training_file, n, r, testing_data):
    cmd = ['java', '-jar', './negative-selection/negsel2.jar', '-self', training_file, '-n', str(n), '-r', str(r), '-c', '-l']
    in_data = bytes('\n'.join(testing_data), encoding="raw_unicode_escape") + b'\x1a'
    p = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    p.stdin.write(in_data)
    return [float(s) for s in p.communicate()[0].decode().splitlines()]

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

    sensitivities = [1] #ensure start at 0,0 and 1,1
    specificities = [0] #ensure start at 0,0 and 1,1
    for i, j in distinct_scores:
        score = sorted_data[i][2]
        TP = len([w for (l, w, s) in sorted_data if s >= score and l])
        FP = len([w for (l, w, s) in sorted_data if s >= score and not l])
        TN = len([w for (l, w, s) in sorted_data if s < score and not l])
        FN = len([w for (l, w, s) in sorted_data if s < score and l])
        TPR = TP / (TP+FN)
        TNR = TN / (TN + FP)

        sensitivities.append(TPR)
        specificities.append(TNR)

    #calculate AUC and make plots
    inverse_spec = np.ones_like(specificities) - specificities
    sensitivities.append(0) #ensure start at 0,0 and 1,1
    inverse_spec = np.append(inverse_spec, 0) #ensure start at 0,0 and 1,1
    auc = metrics.auc(inverse_spec, sensitivities)

    labels = np.array([d[0] for d in data])
    scores = np.array([d[2] for d in data])
    auc_metrics = metrics.roc_auc_score(labels, scores)
    print(f"auc delta: {auc - auc_metrics}")

    plt.figure()
    plt.plot(inverse_spec, sensitivities)
    plt.plot([0.0,1.0], [0.0,1.0], 'r--')
    plt.xlabel('False positive rate (1-specificity)')
    plt.ylabel('True positive rate (Sensitivity)')
    plt.title(f'ROC curve (AUC = {auc})')
    plt.savefig(savename)
    plt.close()


    return auc

if __name__ == '__main__':
    training_file = "./negative-selection/english.train"

    n = 10
    r = 4

    for r in list(range(1,10)):
        testing_english_words = None
        testing_english_labels = None
        testing_english_scores = None

        with open('negative-selection/english.test', 'r') as f:
            (testing_english_labels, testing_english_words) = zip(*[(False, s.strip()) for s in f.readlines()])
            testing_english_scores = run_negative_selection(training_file, n, r, testing_english_words)

        testing_tagalog_words = None
        testing_tagalog_labels = None
        testing_tagalog_scores = None


        for lan in ["hiligaynon", "middle-english", "plautdietsch", "xhosa"]:
            with open('negative-selection/lang/{}.txt'.format(lan), 'r') as f:
                (testing_tagalog_labels, testing_tagalog_words) = zip(*[(True, s.strip()) for s in f.readlines()])
                testing_tagalog_scores = run_negative_selection(training_file, n, r, testing_tagalog_words)

            # with open('negative-selection/tagalog.test', 'r') as f:
            #   (testing_tagalog_labels, testing_tagalog_words) = zip(*[(True, s.strip()) for s in f.readlines()])
            #   testing_tagalog_scores = run_negative_selection(training_file, n, r, testing_tagalog_words)

            english_data = list(zip(testing_english_labels, testing_english_words, testing_english_scores))
            tagalog_data = list(zip(testing_tagalog_labels, testing_tagalog_words, testing_tagalog_scores))

            auc = get_AUC_from_anomalies(english_data + tagalog_data, f"lang-tagalog_n-{n}_r-{r}.png")
            print("The ({}) AUC for R {} is {}".format(lan, r, auc))
