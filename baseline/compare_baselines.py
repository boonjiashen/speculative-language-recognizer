"""Train some baseline classifiers on labeled sentences, compare classification
performance on test data.

The input file should have lines each in the format: INT WORD1 WORD2 WORD3...
where INT is 0 for a non-speculative phrase and 1 for a speculative phrase.
WORD1 WORD2 etc are the corresponding phrase that has been tokenized and
separated by spaces. These tokens may be words or punctuation.
"""

import numpy as np
import argparse
import codecs
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

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
    Strain, Stest, ytrain, ytest = train_test_split(sentences, y,
            test_size=test_proportion, random_state=seed)


    ######################## Define baseline algorithms ####################### 

    named_pipelines = [
            # Define a Naive Bayes classifier that uses TF-IDF
            ('Freq of words -> TDIDF -> NB', Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB()),
                ])),
            # SVM classifier
            ('Freq of words -> TDIDF -> SVM', Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)),
                ])),
            # SVM classifier
            ('Presence of words -> SVM', Pipeline([
                ('vect', CountVectorizer(binary=True)),
                ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)),
                ])),
            # Define a Naive Bayes classifier which counts presence of words
            # rather than the frequency of words
            ('Presence of words -> NB', Pipeline([
                ('vect', CountVectorizer(binary=True)),
                ('clf', MultinomialNB()),
                ])),
            # Define a Naive Bayes classifier with no TF-IDF
            ('Freq of words -> NB', Pipeline([
                ('vect', CountVectorizer()),
                ('clf', MultinomialNB()),
                ])),
            ]

    ######################### Train and test each algorithm ###################

    for pipeline_name, pipeline in named_pipelines:

        # Train classifier
        clf = pipeline.fit(Strain, ytrain)

        # Test classifier
        predictions = clf.predict(Stest)

        # Calculate performance metrics
        classification_report = metrics.classification_report(ytest, predictions)
        accuracy = metrics.accuracy_score(ytest, predictions)
        f1 = metrics.f1_score(ytest, predictions)

        # Width of 1st column for printing
        field_width = max(map(len, zip(*named_pipelines)[0]))

        # Print performance metrics
        print 'Pipeline:', \
            (('%-' + str(field_width) + 's') % pipeline_name),  \
            'F1 score = %f' % f1

