"""Cross-validate a bigram linear SVM classifier to tune hyperparameters.

The input file should have lines each in the format: INT WORD1 WORD2 WORD3...
where INT is 0 for a non-speculative phrase and 1 for a speculative phrase.
WORD1 WORD2 etc are the corresponding phrase that has been tokenized and
separated by spaces. These tokens may be words or punctuation.
"""

import numpy as np
import argparse
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import sklearn.cross_validation

if __name__ == "__main__":

    ######################## Parse command-line arguments ##################### 

    parser = argparse.ArgumentParser()

    # Add argument for more verbose stdout
    parser.add_argument("-v", "--verbose",
            help="print status during program execution", action="store_true")

    # Add required argument of training data
    parser.add_argument('filename', metavar='filepath', type=str,
            help='pre-processed data file containing labeled sentences')

    # Grab arguments from stdin
    args = parser.parse_args()

    # Filename of training data
    filename = args.filename


    ######################## Load sentences and labels ########################

    sentences, y = [], []
    with codecs.open(filename, 'r', 'utf-8') as fid:

        for line in fid:

            space_pos = line.index(' ')  # position of first blank space
            label = int(line[:space_pos])
            sentence = line[space_pos + 1:].strip()

            y.append(label)
            sentences.append(sentence)

    # Split into training and test set
    # S stands for untokenized sentences. I know, it's ugly.
    seed = 0  # seed for random number generator
    test_proportion = 0.3  # proportion of entire dataset we devote to testing
    Strain, Stest, ytrain, ytest =  \
            sklearn.cross_validation.train_test_split(
            sentences, y, test_size=test_proportion, random_state=seed)


    ######################## Define algorithm ################################# 

    # Express a sentence as a bag of bigrams, either the bigram exists (1) or
    # it doesn't (0)
    vectorizer_name = 'Presence of words bigram'
    vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True)

    # Define classification method
    linear_SVM = svm.LinearSVC(C=1)

    # Create classifier pipeline.
    # A pipeline is 1 or more transforms on a sentence, followed by a
    # classification.
    pipeline = Pipeline([
        (vectorizer_name, vectorizer),
        ('SVM', linear_SVM),
        ])


    ######################### Tune hypermeters with CV ########################

    # Get mean of K-folds CV accuracy for different C parameters of SVM
    #parameters = {pipeline.steps[-1][0] + '__C': (1, 10, 1000, 10000)}
    parameters = {
            'SVM__C': (0.1, 1, 10, ),
            'SVM__loss': ('l1', 'l2', ),
            }
    gs_clf = GridSearchCV(pipeline, parameters, cv=3, scoring='f1')
    gs_clf.fit(Strain, ytrain)

    # Print scores of each set of parameters
    for scores in gs_clf.grid_scores_: print scores


    ################### Retrain full train set with best parameters ############

    # Train full training set with best parameters found above
    best_parameters, _, _ = max(
            gs_clf.grid_scores_, key=lambda x: x.mean_validation_score)
    pipeline.set_params(**best_parameters)
    pipeline.fit(Strain, ytrain)

    # Test on test set
    predictions = pipeline.predict(Stest)

    # Calculate performance metrics
    f1 = metrics.f1_score(ytest, predictions)

    # Print performance metrics
    print "Best parameters found over grid search:"
    for param, value in best_parameters.iteritems():
        print param, '=', value
    print 'F1 score = %f' % f1

