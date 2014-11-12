# Try out gensim's word2vec functionality

import gensim
import numpy as np
import scipy.cluster.vq  # for K-means clustering of word vectors
import nltk  # to split plain text into sentences
import itertools
from try_word2vec import lower_tokenized_sentences, GutenbergSentences


if __name__ == "__main__":

    verbose = True  # True if you want to print stages that the script is at during
                    # run time

    # Get sentences from all books in the NLTK corpus, should be about 100K
    # sentences
    filenames = nltk.corpus.gutenberg.fileids()[:1]
    #labeled_sentences = [gensim.models.doc2vec.LabeledSentence([word.lower()
            #for word in sentence], [str(ind)])
            #for ind, sentence in enumerate(GutenbergSentences(filenames))]
    nltk_path = '/Users/bjiashen/nltk_data/corpora/brown'
    #labeled_sentences = [i for i in
            #gensim.models.doc2vec.LabeledBrownCorpus(nltk_path)]

    labeled_sentences = [gensim.models.doc2vec.LabeledSentence(words, label)
            for words, label in [
                (['the', 'cat'], ['label1']),
                (['the', 'dog'], ['label2']),
                (['eat', 'dog'], ['label3']),
                (['eat', 'cat'], ['label4']),
                (['fat', 'mos'], ['label5']),
                ]
            ]

    if verbose:
        print('Training with %i novel%s from Gutenberg...' %  \
                (len(filenames), 's' if len(filenames) > 1 else ''))

    # Train word2vec model
    model = gensim.models.Doc2Vec(labeled_sentences,
            size=5,
            min_count=0)

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
