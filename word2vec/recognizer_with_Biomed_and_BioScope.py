"""Run training and prediction on a speculative language recognizer, given
pre-processed BioScope data (labeled) and Biomed data (unlabeled)

The input file should have lines each in the format: INT WORD1 WORD2 WORD3...
where INT is 0 for a non-speculative phrase and 1 for a speculative phrase.
WORD1 WORD2 etc are the corresponding phrase that has been tokenized and
separated by spaces. These tokens may be words or punctuation.

TERMS
--------------------

id
    e.g. 'SENT_0071'; unique identifier of a sentence. doc2vec calls this a
    'label'

class
    the ground truth classification of a sentence; a binary value

<LS|X|y><train|test>
    e.g. LStrain, ytest
    LS is a list of LabeledSentence objects (i.e. sentences with identifiers)
    X is a list of sentence/phrase vectors
    y is a list of classes, each 0 or 1.
    The recognizer takes a sentence and predicts y. X is the intermediate
    computation.

LStrain_classy
    LabeledSentences from BioScope. These have class labels.
LStrain_classless
    LabeledSentences from Biomed. These have no class labels.

"""

import logging
import os
import gensim
import numpy as np
import argparse
import random
import sklearn.tree
import sklearn.linear_model
import sklearn.metrics
import utils
from Word2VecScorer import Word2VecScorer
from Biomed_word2vec import yield_file_contents, yield_tokenized_sentences

def print_confusion_matrix(y_true, y_pred):
    "Print the confusion matrix of a binary classifier"

    # y_true and y_pred has to contain 0 and 1 and ONLY 0 and 1
    unique_values = set([val for vals in [y_true, y_pred] for val in vals])
    assert len(unique_values) == 2
    assert 0 in unique_values and 1 in unique_values

    confusion_matrix = sklearn.metrics.confusion_matrix(
            y_true, y_pred)  # , labels=target_names)

    # Convert confusion matrix into a table with row and col labels
    cm = confusion_matrix
    table = [
            ['',     'labeled_0',   'labeled_1'],
            ['is_0', str(cm[0, 0]), str(cm[0, 1])],
            ['is_1', str(cm[1, 0]), str(cm[1, 1])],
            ]

    # Measure largest field width in table, for printing
    field_width = max([len(elem) for row in table for elem in row])

    # Print table
    for row in table:
        elements = [(('%' + str(field_width) + 's') % elem) for elem in row]
        print ' '.join(elements)
    

def load_preprocessed_BioScope(filename):
    "Load classes and tokenized sentences from file"

    min_len = 5  # min length of phrase to be trained
                 # For some reason doc2vec cannot train phrases with less
                 # than 5 words/punctuation
    classes, tokenized_sentences = [], []
    with open(filename, 'r') as fid:
        for line in fid:
            tokens = line.split()

            # Check that sentence is long enough
            phrase_len = len(tokens) - 1
            if phrase_len < min_len: continue

            # Extract tokenized sentence (all tokens after first one)
            sentence = tokens[1:]
            tokenized_sentences.append(sentence)

            # Extract class of sentence (first token of line)
            assert tokens[0] in ['0', '1']
            class_ = int(tokens[0])
            classes.append(int(tokens[0]))

    return classes, tokenized_sentences


if __name__ == "__main__":

    logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO)

    random.seed(0)

    ######################## Parse command-line arguments ##################### 

    parser = utils.get_parser()

    # Add required argument of training data
    parser.add_argument('BioScope_file', type=str,
            help='pre-processed data file containing labeled sentences')

    # Add required argument of training data
    parser.add_argument('Biomed_dir', type=str,
            help='directory of Biomed articles')

    # No. of Biomed articles we want to train on
    parser.add_argument('n_Biomed_articles', metavar='n_articles', type=int,
           help='number of Biomed articles to be used as training data')

    # Grab arguments from stdin
    args = parser.parse_args()

    # Check that the expected arguments are in args
    expected_args = ['min_count', 'verbose', 'debug']
    assert set(expected_args) <= set(args.__dict__)

    # Convert parsed inputs into local variables
    locals().update(args.__dict__)
    min_word_count = min_count


    ################### Load Biomed articles ##################################

    articles = []  # article textblocks go here
    src_dir = Biomed_dir  # location of Biomed articles

    # List files in user-given directory
    xml_filenames = os.listdir(src_dir)

    # Choose n_articles at random
    random.shuffle(xml_filenames)
    xml_filenames = xml_filenames[:n_Biomed_articles]

    # Read in articles one by one
    articles = yield_file_contents(xml_filenames, directory=src_dir)

    # Tokenized sentences is an object that can be called more than once
    Biomed_sentences = list(yield_tokenized_sentences(articles))

    n_unlabeled_examples = len(Biomed_sentences)


    ######################### Construct labeled dataset ####################### 

    classes, BioScope_sentences = load_preprocessed_BioScope(BioScope_file)

    # Decide proportion of BioScope that goes to the test set
    test_proportion = 0.3  # Proportion given to test set
                           # The rest goes to training set

    # Generate indices of training and test set
    n_labeled_examples = len(BioScope_sentences)
    inds = range(n_labeled_examples)
    random.shuffle(inds)
    test_inds = inds[:int(test_proportion * n_labeled_examples)]
    train_inds = inds[int(test_proportion * n_labeled_examples):]

    # Function to map the index of an example to its identifer
    index2id = lambda index: 'SENT_%08i' % index

    # Create shortcut variable name for LS
    LabeledSentence = gensim.models.doc2vec.LabeledSentence

    # Split labeled dataset into training and test sets
    LStrain_classy, ytrain, LStest, ytest = [], [], [], []
    for LSlist, ylist, inds in [
            (LStest, ytest, test_inds),
            (LStrain_classy, ytrain, train_inds)]:
        for ind in inds:

            # Construct labeled sentence given the index in the dataset
            sentence = BioScope_sentences[ind]
            id_ = index2id(ind)
            LS = LabeledSentence(sentence, [id_])

            # Get class of this sentence
            class_ = classes[ind]

            # Push to either training or test set
            LSlist.append(LS)
            ylist.append(class_)

    # Cast Biomed sentences into LS datatype
    # Indices continue on from the labeled dataset above
    inds = range(n_labeled_examples, n_labeled_examples + n_unlabeled_examples)
    LStrain_classless = [
            LabeledSentence(sentence, [index2id(ind)])
            for sentence, ind in zip(Biomed_sentences, inds)]

    # Make sure that no. of examples doesn't exceed 6 digits
    # otherwise we need to expand the no. of identifiers of the sentences
    assert n_labeled_examples + n_unlabeled_examples < 10**8


    ######################### Train doc2vec model ############################# 

    # Initialize word model (no training here)
    word_vector_length = 52  # Length of a single word vector
    model = gensim.models.Doc2Vec(
            size=word_vector_length,
            min_count=min_word_count,
            workers=2)

    # Train word2vec model
    model.build_vocab(LStrain_classy + LStrain_classless + LStest)
                        # Build vocab using entire dataset
                        # Without the identifiers of the test set we cannot run
                        # prediction on the test set. Strange API behavior.

    # Train doc2vec model until F1 score on logistic classifier drops
    # We use the classifier as a surrogate for a validation set
    MAX_EPOCHS = 1000  # maximum training epochs
    if verbose:
        print 'Training with %i Biomed sentences and %i BioScope sentences' %  \
                (n_unlabeled_examples, n_labeled_examples)
    clf = sklearn.linear_model.LogisticRegression()
    prev_score = -1
    for ei in range(MAX_EPOCHS):

        # We want the learning rate to be small to ensure the model is
        # converging
        model.alpha = model.min_alpha = 0.01
        model.train(LStrain_classy + LStrain_classless)

        # See how well the doc2vec model does on the logistic classifier
        Xtrain = np.vstack(
                [model[sentence.labels[0]] for sentence in
                LStrain_classy])
        clf.fit(Xtrain, ytrain)
        score = sklearn.metrics.f1_score(ytrain, clf.predict(Xtrain))

        print ('After %i epochs, score is' % (ei + 1)), score

        # Stop training once the score just goes past its peak
        if score < prev_score:
            break

        prev_score = score

    n_test_epochs = ei + 1


    #################### Train classifier #####################################

    clf = sklearn.linear_model.LogisticRegression()
    Xtrain = np.vstack([model[sentence.labels[0]]
        for sentence in LStrain_classy])
    clf.fit(Xtrain, ytrain)


    ######################### Test time: predict test sentences ############### 

    # Generate vector representation of test sentences
    model.train_words = False  # Freeze weights of word vector learning

    n_test_epochs = 50
    if verbose:
        print 'Testing over', n_test_epochs, 'epochs'
    for ei in range(n_test_epochs):
        #model.alpha = model.min_alpha = 0.01
        model.train(LStest)

    # Construct matrix of test sentences
    Xtest = np.vstack([model[sentence.labels[0]] for sentence in LStest])

    # Predict speculative or not
    predictions = clf.predict(Xtest)


    ######################### Print classification metrices ###################

    target_names = ['non-speculative', 'speculative']
    classification_report = sklearn.metrics.classification_report(
            ytest, predictions, target_names=target_names)
    metric_funs = [sklearn.metrics.accuracy_score,
                sklearn.metrics.f1_score,
                sklearn.metrics.precision_score,
                sklearn.metrics.recall_score,
                ]
    metrics = [fun(ytest, predictions) for fun in metric_funs]

    if verbose:
        metric_names = [fun.__name__.split('_')[0] for fun in metric_funs]
        metrics_as_string = ' | '.join([name + ' = ' + ('%.3f' % metric)
                for name, metric in zip(metric_names, metrics)])
        print 'Classification method:', clf.__class__
        print metrics_as_string
        print classification_report
        print 'Confusion matrix:'
        print_confusion_matrix(ytest, predictions)

        print 'Trained with %i Biomed sentences and %i BioScope sentences' %  \
                (n_unlabeled_examples, n_labeled_examples)
