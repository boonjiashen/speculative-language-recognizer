# Learn fixed length feature vectors for words from the BioScope corpus
# Learning makes use of the gensim library

import gensim
import numpy as np
import itertools
import nltk
from bs4 import BeautifulSoup


class get_tokenized_sentences_from_Bioscope(object):
    """Generate tokenized sentences given BioScope XML filenames.
    
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

    verbose = True  # True if you want to print stages that the script is at during
                    # run time

    # Filenames of BioScope XML files
    filenames = ["../data/abstracts.xml", "../data/abstracts_pmid.xml",
            "../data/full_papers.xml"]

    # Get tokenized sentences from BioScope corpus
    tokenized_sentences = get_tokenized_sentences_from_Bioscope(filenames)

    get_sentences = lambda: tokenized_sentences;

    if verbose:
        print 'Training model with %i sentences...' %  \
                len([i for i in get_sentences()])

    # Train word2vec model
    model = gensim.models.Word2Vec(
            size=50,
            min_count=5)
    model.build_vocab(get_sentences())
    model.train(get_sentences())
