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
from sklearn import svm

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

    # Define several classification methods
    linear_SVM = svm.LinearSVC(C=1)
    RBF_SVM = svm.SVC(kernel='rbf')
    NB_method = MultinomialNB()
    named_clf_methods = [
            ('linear SVM', linear_SVM),
            #('SVM with RBF kernel', RBF_SVM),
            ('NB', NB_method)]

    # Each named pipeline is a (description, pipeline) 2-ple.
    # Each pipeline is 1 or more transforms on a sentence, followed by a
    # classification.
    named_pipelines = []
    for binarize_word_counts in [True, False]:  # Word count or word presence
        for clf_name, clf_method in named_clf_methods:
            for n_grams in [1, 2]:  # Unigrams or bigrams

                # Create the list of transforms and the classifier at the end
                named_transforms = []

                # Add either word count or word presence
                vectorizer_name = ('Presence of words'  \
                        if binarize_word_counts else 'Freq of words') +  \
                        ' ' +  \
                        (str(n_grams) + '-grams')
                vectorizer = CountVectorizer(
                        ngram_range=(1, n_grams),
                        binary=binarize_word_counts)
                named_transforms.append((vectorizer_name, vectorizer, ))

                # Add classification method at the tail
                named_transforms.append((clf_name, clf_method,))

                # Derive name by joining names of all the transforms
                name = ' -> '.join(zip(*named_transforms)[0])

                # Append to the list of pipelines
                named_pipeline = (name, Pipeline(named_transforms))
                named_pipelines.append(named_pipeline)


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
            ' | F1 score = %f' % f1
