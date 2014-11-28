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


if __name__ == "__main__":


    ######################## Parse command-line arguments ##################### 

    parser = argparse.ArgumentParser()

    # Add argument for more verbose stdout
    parser.add_argument("-v", "--verbose",
            help="print status during program execution", action="store_true")

    # No. of articles we want to download
    parser.add_argument('n_articles', metavar='n_articles', type=int,
           help='number of Biomed articles to be used as training data')

    # Grab arguments from stdin
    args = parser.parse_args()

    # Move cmd arguments to be local variables
    locals().update(args.__dict__)


    ################### Load sentences ########################################

    # Read list of XML filenames
    filenames_file = 'data/Biomed_filenames'  # File storing filenames
    with open(filenames_file) as fid:
        xml_filenames = fid.read().splitlines()

    if verbose:
        sys.stdout.write(
            'Downloading %i Biomed articles... ' % n_articles)
        sys.stdout.flush()

    # Download a couple of Biomedical articles and grab their sentences
    p_texts = []  # text of <p> tags from Biomed XML files
                    # Most are single sentences but some have more than one
    ftp = utils.get_Biomed_FTP_object()  # Login to FTP and cd to appropriate dir
    for filename in xml_filenames[:n_articles]:
        textblock = utils.get_Biomed_XML_as_string(ftp=ftp, src_filename=filename)
        curr_sentences = utils.retrieve_sentences_from_Biomed_textblock(textblock)

        p_texts.extend(curr_sentences)

    ftp.close()  # Close FTP connection

    if verbose: sys.stdout.write('done.\n')

    ######################### Train model ##################################### 

    # Get tokenized sentences from Biomed articles
    tokenized_sentences = [[word.lower() for word in nltk.word_tokenize(sentence)]
            for sentence in p_texts]

    get_sentences = lambda: tokenized_sentences;

    if verbose:
        sys.stdout.write('Training model with %i sentences... ' %  \
                len([i for i in get_sentences()]))
        sys.stdout.flush()

    # Train word2vec model
    word_vector_length = 50  # Length of a single word vector
    min_word_count = 5  # Min count to allow a word in the vocabulary
    model = gensim.models.Word2Vec(
            size=word_vector_length,
            min_count=min_word_count)
    model.build_vocab(get_sentences())
    model.train(get_sentences())

    if verbose: sys.stdout.write('done.\n')
    

    ########################## Test model qualitatively ######################## 

    #test_words = ['blood', 'demonstrated']
    #for test_word in test_words:
        #if test_word not in model.vocab:
            #print 'Test word "%s" not in vocab' % test_word
            #continue

        ## Get test_words similar to this test test_word
        #similar_words = [word
                #for word, similarity in model.most_similar(test_word)]

        ## Print similar words
        #print 'Words similar to "%s":' % test_word
        #print '\t', ' '.join(similar_words)
