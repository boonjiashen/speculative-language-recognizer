"""Simple script to plot precision recall curves given a file of labels and
confidences

We expect the format of the input file to be the following format
Line #1: list of labels (0s and 1s) for curve 1
Line #2: list of confidences (higher means more confident in predicting 1) for
         curve 1
Line #3: list of labels for curve 2
Line #4: list of confidences for curve 2
etc.
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import sklearn.metrics

parser = argparse.ArgumentParser()

parser.add_argument('input_filename', type=str)

args = parser.parse_args()

data = np.loadtxt(args.input_filename)

plt.figure()

for ci in range(0, len(data), 2):
    labels, confidences = data[ci:ci+2]
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels,
            confidences)
    label = 'Lines %i and %i' % (ci+1, ci+2)
    plt.plot(recall, precision, label=label)

plt.xlim(xmin=0)  # set axes to cut through origin
plt.ylim(ymin=0)
plt.xlabel('Recall')
plt.ylabel('Precision')

# Show legend only if there's more than one curve
if len(data) > 2:
    plt.legend(loc='best')

plt.show()
