# Learn fixed length feature vectors for words from the BioScope corpus
# Learning makes use of the gensim library

import gensim
import numpy as np
import itertools
import nltk
from bs4 import BeautifulSoup


def get_sentences_from_Bioscope(filename):
    "Return a list of sentences given a BioScope XML filename."

    # Load a list of sentences
    fid = open(filename)
    textblock = fid.read()
    fid.close()
    soup = BeautifulSoup(textblock)
    sentences = [tag.get_text() for tag in soup.find_all('sentence')]

    return sentences

if __name__ == "__main__":

    verbose = True  # True if you want to print stages that the script is at during
                    # run time

    # Filenames of BioScope XML files
    filenames = ["../data/abstracts.xml", "../data/abstracts_pmid.xml",
            "../data/full_papers.xml"]

    # Get sentences from BioScope corpus
    sentences = get_sentences_from_Bioscope('../data/abstracts.xml')



    # Tokenize sentences into words and make lower case
    tokenized_sentences = [
            [word.lower() for word in nltk.word_tokenize(sentence)]
            for sentence in sentences]

    get_sentences = lambda: tokenized_sentences;

    if verbose:
        print 'Training', len(tokenized_sentences), 'sentences'

    # Train word2vec model
    model = gensim.models.Word2Vec(
            size=50,
            min_count=5)
    model.build_vocab(get_sentences())
    model.train(get_sentences())

    assert False

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
