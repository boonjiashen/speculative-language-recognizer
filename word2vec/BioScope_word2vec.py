# Learn fixed length feature vectors for words from the BioScope corpus
# Learning makes use of the gensim library

import gensim
import numpy as np
import itertools
import nltk
import argparse
from bs4 import BeautifulSoup

class Word2VecScorer(object):


    def __init__(self, model):
        "Takes in a word2vec or doc2vec model"
        self.comparisons = []  # a list of 4-ples
        self._build_comparisons(model)

    def _build_comparisons(self, model):
        """Build the list of comparisons appropriate for this model.

        We only make use of grammar comparisons since it's the only categories
        where our training data has the required vocabulary. We also ignore
        comparisons if any of the four items in the comparison are not in the
        model's vocabulary.
        """

        # Each line is a comparison 'x y X Y' i.e. x is to y what X is to Y EXCEPT
        # for categories, which are lines that start with a colon, e.g.
        #: city-in-state
        #: family
        #: gram1-adjective-to-adverb
        #: gram2-opposite
        evaluation_filename = 'data/questions-words.txt'
        with open(evaluation_filename) as fid:
            lines = fid.read().splitlines()

        # Find out where the grammar categories start.
        start_ind = [ind for ind, line in enumerate(lines)
                if line[0] == ':' and 'gram' in line][0]

        # Grab all grammar comparisons as a list of 4-ples
        # We happen to know all the grammar comparisons are at the end of the file.
        vocab = set(model.vocab.keys())
        for line in lines:

            # Skip category headers
            if line[0] == ':':
                continue

            # Don't bother testing if any of these words aren't in vocab
            tokens = line.split()
            if not set(tokens) <= vocab:
                continue

            self.comparisons.append(tokens)

    def score(self, model, topn, return_correct_comparisons=False):
        """Score a model by how many correct comparisons it makes from the
        set of analogies

        topn - correct answer should be within top #N predictions
        """

        # Start comparisons
        correct_inds = []  # Inds of comparisons that were predicted right
        for ci, comparison in enumerate(self.comparisons):

            # E.g. 'stupid clever slow fast'
            # which implies stupid is to clever what slow is to fast.
            # We want to predict 'fast' from the first three words
            pos1, neg1, pos2, neg2 = comparison

            try:

                # Make predictions (these come with confidence levels)
                predictions_and_confidence = model.most_similar(
                        positive=[pos1, pos2], negative=[neg1], topn=topn)
                predictions = zip(*predictions_and_confidence)[0]

                # Update performance metrics
                if neg2 in predictions:
                    correct_inds.append(ci)
            except KeyError:
                continue

        score = 1. * len(correct_inds) / len(self.comparisons)
        correct_comparisons = [self.comparisons[i] for i in correct_inds]

        if return_correct_comparisons:
            return score, correct_comparisons
        else:
            return score

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
    model.build_vocab(get_sentences())
    model.train(get_sentences())
    

    ######################### Test model qualitatively ######################## 
    
    scorer = Word2VecScorer(model)
    for topn in [1, 2, 3]:
        print 'Top', topn, scorer.score(model, topn)
