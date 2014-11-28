# Learn fixed length feature vectors for words from the BioScope corpus
# Learning makes use of the gensim library

import gensim
import numpy as np
import itertools
import nltk
import argparse
from bs4 import BeautifulSoup
from Word2VecScorer import Word2VecScorer

class get_tokenized_sentences_from_Bioscope(object):
    """Generate tokenized sentences given BioScope XML filenames.
    
    Each tokenized sentence is a list of words, e.g. ['I', 'ate', 'food'] is a
    tokenized sentence.
    >>> for sentence in get_sentences_from_Bioscope(['abstracts.xml']):
    ...     print sentence
    """

    def __init__(self, filenames):
        self.filenames = filenames

    def __iter__(self):

        for filename in self.filenames:

            # Load a list of sentences
            fid = open(filename)
            textblock = fid.read()
            fid.close()
            soup = BeautifulSoup(textblock)
            sentences = [tag.get_text() for tag in soup.find_all('sentence')]

            # Yield sentence by sentence
            for sentence in sentences:
                yield [word.lower() for word in nltk.word_tokenize(sentence)]


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
    filenames = ["../data/abstracts.xml", "../data/full_papers.xml"]
    filenames = ["../data/one_abstract.xml"]
    filenames = args.filenames


    ######################### Train model ##################################### 

    # Get tokenized sentences from BioScope corpus
    tokenized_sentences = get_tokenized_sentences_from_Bioscope(filenames)

    #tokenized_sentences = [x for x in tokenized_sentences]
    #tokenized_sentences = tokenized_sentences[:1000]

    get_sentences = lambda: tokenized_sentences;

    if args.verbose:
        print 'Training model with %i sentences...' %  \
                len([i for i in get_sentences()])

    # Train word2vec model
    word_vector_length = 50  # Length of a single word vector
    min_word_count = 5  # Min count to allow a word in the vocabulary
    model = gensim.models.Word2Vec(
            size=word_vector_length,
            min_count=min_word_count)

    # Build vocabulary
    model.build_vocab(get_sentences())

    # Create scorer based in this model
    scorer = Word2VecScorer(model)

    # Get score for each training epoch
    n_epochs = 50
    for ei in range(n_epochs):

        # Train for one epoch
        model.train(get_sentences())

        # Evaluate model after each epoch
        scores = [scorer.score(model, topn) for topn in [1, 2, 3, 4, 5]]
        print ('After %i epochs, score is' % (ei + 1)), scores
