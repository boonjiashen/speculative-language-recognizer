"""Simple script to plot precision recall curves given a file of instance
labels and confidences

The number of lines in the the input file should be in multiples of 3.
Line #1: Name of curve 1
Line #2: list of labels (0s and 1s) for curve 1
Line #3: list of confidences (higher means more confident in predicting 1) for
         curve 1
Line #4: Name of curve 2
etc.
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import sklearn.metrics
import StringIO

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('input_filename', type=str)

    args = parser.parse_args()


    #################### Read lines in file ###################################

    with open(args.input_filename, 'r') as fid:

        # Read in lines and skip over empty lines
        lines = [x for x in fid.read().splitlines() if x.strip()]

    plt.figure()


    #################### Plot PR curves #######################################

    for ci in range(0, len(lines), 3):
        curve_name, labels, confidences = lines[ci:ci+3]

        # Remove leading and trailing spaces from curve name
        curve_name = curve_name.strip()

        # Parse strings into NumPy arrays
        labels = [int(x) for x in labels.strip().split()]
        confidences = np.loadtxt(StringIO.StringIO(confidences))

        # Get precision, recall curve and plot it
        precision, recall, _ = sklearn.metrics.precision_recall_curve(labels,
                confidences)
        plt.plot(recall, precision, label=curve_name)

    plt.legend(loc='best')
    plt.xlim(xmin=0)  # set axes to cut through origin
    plt.ylim(ymin=0)
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.show()
