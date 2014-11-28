# Try out gensim's word2vec functionality

import gensim
import numpy as np
import scipy.cluster.vq  # for K-means clustering of word vectors
#from bs4 import BeautifulSoup  # to parse XML
import nltk  # to split plain text into sentences
import itertools
import utils


#def get_sentences_from_Bioscope(filename):
    #"Return a list of sentences given a BioScope XML filename."

    ## Load a list of sentences
    #fid = open(filename)
    #textblock = fid.read()
    #fid.close()
    #soup = BeautifulSoup(textblock)
    #sentences = [tag.get_text() for tag in soup.find_all('sentence')]

    #return sentences


def get_sentences_from_plain_text_file(filename):
    "Return a list of sentences given a plan text file"

    # Get a list of sentences from a plain text file
    # Source:
    # http://stackoverflow.com/questions/4576077/python-split-text-on-sentences
    fid = open(filename)
    textblock = fid.read()
    fid.close()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(textblock.lower())

    return sentences


def lower_tokenized_sentences(generator):
    """Makes sentences of a sentence generator lowercase

    Assumes that the input generator generates sentences, each sentence being a
    list of words.
    """
    for sentence in generator:
        yield [word.lower() for word in sentence]

#sentences = get_sentences_from_Bioscope('../data/abstracts.xml')
#sentences = get_sentences_from_plain_text_file('data/austen-emma.txt')

@utils.multigen
def yield_gutenberg_sentences(filenames):
    """Generator object that yields lists of words from novels in the NLTK lib 

    Takes in a list of filenames recognized by ntlk.corpus.gutenberg.sents
    >>> # Get generator
    >>> sentences = yield_gutenberg_sentences(['austen-emma.txt', 'austen-persuasion'])
    >>> # Print one list of words per iteration
    >>> for sentence in sentences: print sentence
    """
    for filename in filenames:
        for sentence in nltk.corpus.gutenberg.sents(filename):
            yield sentence


if __name__ == "__main__":

    verbose = True  # True if you want to print stages that the script is at during
                    # run time

    # Get sentences from all books in the NLTK corpus, should be about 100K
    # sentences
    filenames = nltk.corpus.gutenberg.fileids()[:1]
    #get_sentences = lambda: lower_tokenized_sentences(GutenbergSentences(filenames))
    get_sentences = lambda: (
            [word.lower() for word in sentence]
            for sentence in yield_gutenberg_sentences(filenames)
            )

    if verbose:
        print('Training with %i novel%s from Gutenberg...' %  \
                (len(filenames), 's' if len(filenames) > 1 else ''))

    # Train word2vec model
    model = gensim.models.Word2Vec(
            size=50,
            min_count=3)
    model.build_vocab(get_sentences())
    model.train(get_sentences())

    # Word matrix, each row is the vector of a word
    vocab = list(model.vocab.keys())
    word_matrix = np.array([model[word] for word in vocab])

    # K-means cluster all word vectors
    n_clusters = 10
    _, clusters = scipy.cluster.vq.kmeans2(word_matrix, n_clusters)

    # Print size of each cluster
    for ci in range(n_clusters):
        print('%i words in cluster %i' % (sum(clusters == ci), ci))

    # Print some words from each cluster
    print('\nRandom words in each cluster')
    n_words = 10
    for ci in range(n_clusters):

        # Get indices of words in word matrix V
        vis = np.flatnonzero(clusters == ci)[:n_words]
        words = (vocab[vi] for vi in vis)

        print('Cluster %i: %s' % (ci, ' '.join(words)))
