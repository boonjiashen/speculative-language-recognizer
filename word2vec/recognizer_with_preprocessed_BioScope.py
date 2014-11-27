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
import sklearn.metrics
import utils

if __name__ == "__main__":


    ######################## Parse command-line arguments ##################### 

    parser = utils.get_parser()

    # Add required argument of training data
    parser.add_argument('filename', metavar='filepath', type=str,
            help='pre-processed data file containing labeled sentences')

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
    test_proportion = 0.2  # Proportion given to test set
                           # The rest goes to training set
    n_examples = len(tokenized_sentences)
    inds = range(n_examples)
    random.shuffle(inds)
    test_inds = inds[:int(test_proportion * n_examples)]
    train_inds = inds[int(test_proportion * n_examples):]

    # Function to map the index of an example to its identifer
    index2id = lambda index: 'SENT_%06i' % index

    # Split dataset into training and test sets
    LStrain, ytrain, LStest, ytest = [], [], [], []
    LabeledSentence = gensim.models.doc2vec.LabeledSentence
    for to_test, inds in [(True, test_inds), (False, train_inds)]:
        for ind in inds:

            # Construct labeled sentence given the index in the dataset
            sentence = tokenized_sentences[ind]
            id_ = index2id(ind)
            LS = LabeledSentence(sentence, [id_])

            # Get class of this sentence
            class_ = classes[ind]

            # Push to either training or test set
            if to_test:
                LStest.append(LS)
                ytest.append(class_)
            else:
                LStrain.append(LS)
                ytrain.append(class_)

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
    model.train(LStrain)  # train using training set
    

    ######################### Train phrase vector classifier ##################

    # Construct vector representation of sentences
    #Xtrain = np.vstack([model[sentence.labels[0]] for sentence in LStrain])
    vectors = []
    for labeled_sentence in LStrain:
        id_ = labeled_sentence.labels[0]
        vector = model[id_]

        vectors.append(vector)
    Xtrain = np.vstack(vectors)

    # Train decision tree!
    clf = sklearn.tree.DecisionTreeClassifier()
    clf = clf.fit(Xtrain, ytrain)


    ######################### Test time: predict test sentences ############### 

    # Generate vector representation of test sentences
    model.train_words = False  # Freeze weights for word vector learning
    model.train(LStest)

    # Construct matrix of test sentences
    Xtest = np.vstack([model[sentence.labels[0]] for sentence in LStest])

    # Predict speculative or not
    predictions = clf.predict(Xtest)

    ######################### Print classification metrices ###################

    target_names = ['non-speculative', 'speculative']
    classification_report = sklearn.metrics.classification_report(
            ytest, predictions, target_names=target_names)
    accuracy = sklearn.metrics.accuracy_score(ytest, predictions)
    f1 = sklearn.metrics.f1_score(ytest, predictions)

    print 'accuracy = %f | F1 = %f' % (accuracy, f1)
