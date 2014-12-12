"""Run training and prediction on a speculative language recognizer, given
pre-processed BioScope data.

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
"""

import gensim
import numpy as np
import argparse
import random
import sklearn.tree
import sklearn.linear_model
import sklearn.metrics
import utils
from Word2VecScorer import Word2VecScorer

if __name__ == "__main__":

    random.seed(0)

    ######################## Parse command-line arguments ##################### 

    parser = utils.get_parser()

    # Add required argument of training data
    parser.add_argument('filename', metavar='filepath', type=str,
            help='pre-processed data file containing labeled sentences')

    # Learning rate
    default_learning_rate = 0.025
    parser.add_argument('--learning_rate', type=float,
            help='constant learning rate for training (default=%f)'  \
                    % default_learning_rate,
            default=default_learning_rate)

    # Grab arguments from stdin
    args = parser.parse_args()

    # Check that the expected arguments are in args
    expected_args = ['n_epochs', 'min_count', 'verbose', 'debug']
    assert set(expected_args) <= set(args.__dict__)

    # Convert parsed inputs into local variables
    locals().update(args.__dict__)
    min_word_count = min_count


    ######################### Construct datasets ############################## 

    # Load classes and tokenized sentences from file
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

    # Decide which example in dataset goes to training or test set
    test_proportion = 0.3  # Proportion given to test set
    cv_proportion = 0.1    # Proportion given to CV set
                           # The rest goes to training set

    # Generate indices of training, CV and test set
    n_examples = len(tokenized_sentences)
    inds = range(n_examples)
    random.shuffle(inds)
    test_inds = inds[:int(test_proportion * n_examples)]
    cv_inds = inds[int(test_proportion * n_examples):  \
            int((test_proportion + cv_proportion) * n_examples)]
    train_inds = inds[int((test_proportion + cv_proportion) * n_examples):]

    # Function to map the index of an example to its identifer
    index2id = lambda index: 'SENT_%06i' % index

    # Split dataset into training and test sets
    LScv, ycv, LStrain, ytrain, LStest, ytest = [], [], [], [], [], []
    LabeledSentence = gensim.models.doc2vec.LabeledSentence
    for LSlist, ylist, inds in [
            (LScv, ycv, cv_inds),
            (LStest, ytest, test_inds),
            (LStrain, ytrain, train_inds)]:
        for ind in inds:

            # Construct labeled sentence given the index in the dataset
            sentence = tokenized_sentences[ind]
            id_ = index2id(ind)
            LS = LabeledSentence(sentence, [id_])

            # Get class of this sentence
            class_ = classes[ind]

            # Push to either training or test set
            LSlist.append(LS)
            ylist.append(class_)

    # Make sure that no. of examples doesn't exceed 6 digits
    # otherwise we need to expand the no. of identifiers of the sentences
    assert len(tokenized_sentences) < 10**6

    ######################### Train doc2vec model ############################# 

    # Initialize word model (no training here)
    word_vector_length = 50  # Length of a single word vector
    model = gensim.models.Doc2Vec(
            size=word_vector_length,
            min_count=min_word_count)

    # Train word2vec model
    model.build_vocab(LStrain + LStest)  # Build vocab using entire dataset
                        # Without the identifiers of the test set we cannot run
                        # prediction on the test set. Strange API behavior.
                        
    # Create scorer based in this model
    scorer = Word2VecScorer(model)

    # Train doc2vec model
    # As a by-product we'll learn the logistic regressor used to predict
    # labels, since this regressor is used here to monitor the quality of the
    # learnt doc2vec model
    clf = sklearn.linear_model.LogisticRegression()
    for ei in range(n_epochs):

        # Train for one epoch
        model.alpha = model.min_alpha = learning_rate
        model.train(LStrain)

        # Evaluate model after each epoch
        # We score by training the features of the training sentences on
        # a logistic regressor and computing the training error
        Xtrain = np.vstack([model[sentence.labels[0]] for sentence in LStrain])
        score = clf.fit(Xtrain, ytrain).score(Xtrain, ytrain)
        #score = scorer.mean_similarity(model)

        print ('After %i epochs, score is' % (ei + 1)), score
    

    ######################### Test time: predict test sentences ############### 

    # Generate vector representation of test sentences
    model.train_words = False  # Freeze weights for word vector learning

    if verbose:
        print 'Testing over', n_epochs, 'epochs'
    for ei in range(n_epochs):
        model.train(LStest)

    # Construct matrix of test sentences
    Xtest = np.vstack([model[sentence.labels[0]] for sentence in LStest])

    # Predict speculative or not
    predictions = clf.predict(Xtest)


    ######################### Print classification metrices ###################

    target_names = ['non-speculative', 'speculative']
    classification_report = sklearn.metrics.classification_report(
            ytest, predictions, target_names=target_names)
    confusion_matrix = sklearn.metrics.confusion_matrix(
            ytest, predictions)  # , labels=target_names)
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
        print clf.__class__
        print metrics_as_string
        print classification_report
        print confusion_matrix
