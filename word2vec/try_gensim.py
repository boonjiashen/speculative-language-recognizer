# Try out gensim's word2vec functionality

import gensim
import numpy as np
import scipy.cluster.vq  # for K-means clustering of word vectors
from bs4 import BeautifulSoup

# Filename of text file
filename = '../data/abstracts.xml'

def get_sentences_from_Bioscope(filename):
    "Return a list of sentences given a BioScope XML filename."

    # Load a list of sentences
    fid = open(filename)
    textblock = fid.read()
    fid.close()
    soup = BeautifulSoup(textblock)
    sentences = (tag.get_text() for tag in soup.find_all('sentence'))

    return sentences

sentences = get_sentences_from_Bioscope(filename)

# Train word2vec model
model = gensim.models.word2vec.Word2Vec((i.split() for i in sentences),
        size=50,
        min_count=3)

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
n_words = 10
for ci in range(n_clusters):

    # Get indices of words in word matrix V
    vis = np.flatnonzero(clusters == ci)[:n_words]
    words = '\n'.join(vocab[vi] for vi in vis)

    print('Cluster %i:\n%s\n' % (ci, words))
