# Learn fixed length feature vectors for sentences from the BioScope corpus
# Learning makes use of the gensim library
# Useful resources to understand gensim's doc2vec
# https://github.com/piskvorky/gensim/pull/231#issuecomment-59741971

import gensim
import numpy as np
import itertools
import nltk
import argparse
import random
from bs4 import BeautifulSoup
from BioScope_word2vec import get_tokenized_sentences_from_Bioscope

if __name__ == "__main__":


    ######################## Parse command-line arguments ##################### 

    parser = argparse.ArgumentParser()

    # Add argument for more verbose stdout
    parser.add_argument("-v", "--verbose",
            help="print status during program execution", action="store_true")

    # Add required argument of training data
    parser.add_argument('filenames', metavar='filepath', type=str, nargs='+',
                               help='one or more XML files used as training data')

    # Grab arguments from stdin
    args = parser.parse_args()

    # Filenames of BioScope XML files
    filenames = args.filenames


    ######################### Pre-process dataset ############################# 

    # Get tokenized sentences from BioScope corpus
    tokenized_sentences = [i
            for i in get_tokenized_sentences_from_Bioscope(filenames)]

    # Decide which example in dataset goes to training or test set
    test_proportion = 0.2  # Proportion given to test set
                           # The rest goes to training set
    n_examples = len(tokenized_sentences)
    inds = range(n_examples)
    random.shuffle(inds)
    test_inds = inds[:int(test_proportion * n_examples)]
    train_inds = inds[int(test_proportion * n_examples):]

    # Make sure that no. of examples doesn't exceed 6 digits
    # otherwise we need to expand the label of the sentences
    assert len(tokenized_sentences) < 10**6

    # Function to map the index of an example to its label
    ind2label = lambda ind: 'SENT_%06i' % ind

    # Label the dataset
    labeled_sentences = [
        gensim.models.doc2vec.LabeledSentence(sentence, [ind2label(ind)])
        for ind, sentence in enumerate(tokenized_sentences)]


    ######################### Train model ##################################### 

    get_sentences = lambda: labeled_sentences;

    # Initialize word model (no training here)
    word_vector_length = 50  # Length of a single word vector
    min_word_count = 5  # Min count to allow a word in the vocabulary
    model = gensim.models.Doc2Vec(
            size=word_vector_length,
            min_count=min_word_count)

    # Train word2vec model
    train_set = [labeled_sentences[i] for i in train_inds]
    model.build_vocab(labeled_sentences)  # build vocab using entire dataset
    model.train(train_set)  # train using training set
    

    ######################### Get vectors for testset #########################

    if args.verbose:
        print 'Before learning test set:'
        for ind in test_inds:
            sentence_vec = model[ind2label(ind)]
            print sentence_vec

    model.train_words = False  # freeze weights for word vector learning
    test_set = [labeled_sentences[i] for i in test_inds]
    model.train(test_set)

    if args.verbose:
        print 'After learning test set:'
        for ind in test_inds:
            sentence_vec = model[ind2label(ind)]
            print sentence_vec


    ######################### Test model qualitatively ######################## 

    test_words = ['blood', 'demonstrated']
    for test_word in test_words:
        if test_word not in model.vocab:
            print 'Test word "%s" not in vocab' % test_word
            continue

        # Get test_words similar to this test test_word
        similar_words = [word
                for word, similarity in model.most_similar(test_word)]

        # Print similar words
        print 'Words similar to "%s":' % test_word
        print '\t', ' '.join(similar_words)
