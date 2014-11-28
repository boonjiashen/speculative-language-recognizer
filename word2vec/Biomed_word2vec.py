# Learn fixed length feature vectors for words from the Biomed database
#
# The difference between using Biomed and Bioscope is that Biomed contains
# more (unlabeled) data. We hope to get better performance on learning word
# vectors with the larger amount of data, which may be then fed into the
# supervised doc2vec phase.

import gensim
import numpy as np
import itertools
import nltk
import argparse
import utils
import sys
import os
import random
from Word2VecScorer import Word2VecScorer

@utils.multigen
def yield_file_contents(filenames, directory=''):
    """Return contents of files one-by-one as textblocks"""
    for filename in filenames:
        full_path = os.path.join(directory, filename)
        with open(full_path) as fid:
            article = fid.read()
        yield article


if __name__ == "__main__":

    random.seed(0)  # seed RNG to make it deterministic in each run

    ######################## Parse command-line arguments ##################### 

    parser = utils.get_parser()

    # No. of articles we want to download
    parser.add_argument('n_articles', metavar='n_articles', type=int,
           help='number of Biomed articles to be used as training data')

    # Local directory of XML files (if unspecified, download articles from
    # Internet)
    parser.add_argument('--local', dest='src_dir',
           help='local directory of Biomed articles in XML (if ' +  \
                   'unspecified, we download articles from the Internet)')

    # Grab arguments from stdin
    args = parser.parse_args()

    # Check that the expected arguments are in args
    expected_args = ['n_epochs', 'verbose']
    assert set(expected_args) <= set(args.__dict__)

    # Convert parsed inputs into local variables
    locals().update(args.__dict__)
    min_word_count = min_count


    ################### Load articles  ########################################

    # Get articles either from local disk or Internet
    articles = []
    if src_dir:  # Get articles from local disk

        # List files in user-given directory
        xml_filenames = os.listdir(src_dir)

        # Choose n_articles at random
        random.shuffle(xml_filenames)
        xml_filenames = xml_filenames[:n_articles]

        # Read in articles one by one
        articles = yield_file_contents(xml_filenames, directory=src_dir)

    else:  # Get articles from Internet

        # Read list of XML filenames
        filenames_file = 'data/Biomed_filenames'  # File storing filenames
        with open(filenames_file) as fid:
            xml_filenames = fid.read().splitlines()[:n_articles]

        # Print status
        if verbose:
            sys.stdout.write(
                'Downloading %i Biomed articles... ' % n_articles)
            sys.stdout.flush()

        # Login to FTP and cd to appropriate dir
        ftp = utils.get_Biomed_FTP_object()

        # Download Biomed XML as blocks of text
        articles = [
                utils.get_Biomed_XML_as_string(ftp=ftp, src_filename=filename)
                for filename in xml_filenames]

        ftp.close()  # Close FTP connection

        if verbose: sys.stdout.write('done.\n')


    ######################### Parse articles into tokenized sentences #########

    # Get tokenized sentences from Biomed articles
    @utils.multigen
    def yield_tokenized_sentences(articles):
        for article in articles:
            sentences = utils.retrieve_sentences_from_Biomed_textblock(article)
            for sentence in sentences:
                yield [word.lower() for word in nltk.word_tokenize(sentence)]

    # Tokenized sentences is an object that can be called more than once
    tokenized_sentences = yield_tokenized_sentences(articles)

    get_sentences = lambda: tokenized_sentences


    ######################### Train model ##################################### 

    if verbose:
        if isinstance(get_sentences(), list):
            sys.stdout.write('Training model with %i sentences... ' %  \
                    len(get_sentences()))
        else:
            sys.stdout.write('Training model... ')
        sys.stdout.flush()

    # Train word2vec model
    word_vector_length = 50  # Length of a single word vector
    model = gensim.models.Word2Vec(
            size=word_vector_length,
            min_count=min_word_count)

    # Build vocabulary
    model.build_vocab(get_sentences())

    # Create scorer based in this model
    scorer = Word2VecScorer(model)

    # Get score for each training epoch
    learning_rate = 0.002  
    for ei in range(n_epochs):

        # Train for one epoch
        model.alpha = model.min_alpha = learning_rate
        model.train(get_sentences())

        # Evaluate model after each epoch
        score = scorer.mean_similarity(model)
        #scores = [scorer.score(model, topn, percentage=False)
                #for topn in [1, 2, 3, 4, 5]]
        print ('After %i epochs, score is' % (ei + 1)), score

    if verbose: sys.stdout.write('done.\n')
