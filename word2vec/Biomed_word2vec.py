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
from Word2VecScorer import Word2VecScorer


if __name__ == "__main__":


    ######################## Parse command-line arguments ##################### 

    parser = utils.get_parser()

    # No. of articles we want to download
    parser.add_argument('n_articles', metavar='n_articles', type=int,
           help='number of Biomed articles to be used as training data')

    # Grab arguments from stdin
    args = parser.parse_args()

    # Check that the expected arguments are in args
    expected_args = ['n_epochs', 'verbose']
    assert set(expected_args) <= set(args.__dict__)

    # Convert parsed inputs into local variables
    locals().update(args.__dict__)
    min_word_count = min_count


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
    model = gensim.models.Word2Vec(
            size=word_vector_length,
            min_count=min_word_count)


    # Build vocabulary
    model.build_vocab(get_sentences())

    # Create scorer based in this model
    scorer = Word2VecScorer(model)

    def get_mean_similarity():
        "1 number metric on how well a word2vec model is doing"

        model.init_sims()
        def get_norm_vector(word):
            return model.syn0norm[model.vocab[word].index]
        similarities = []  # How well model does for each comparison
        for pos1, neg1, pos2, neg2 in scorer.comparisons:

            # Get offsets pos1-neg1 and pos2-neg2
            # These vectors should be similar in a well-trained model
            offsets = [get_norm_vector(pos) - get_norm_vector(neg)
                    for pos, neg in [(pos1, neg1), (pos2, neg2)]]

            similarity = np.dot(offsets[0], offsets[1])

            similarities.append(similarity)

        mean_similarity = np.mean(similarities) if similarities else 0

        return mean_similarity

    # Get score for each training epoch
    for ei in range(n_epochs):

        # Train for one epoch
        model.alpha = model.min_alpha = 0.001  # Learning rate
        model.train(get_sentences())

        # Evaluate model after each epoch
        score = get_mean_similarity()
        #scores = [scorer.score(model, topn, percentage=False)
                #for topn in [1, 2, 3, 4, 5]]
        print ('After %i epochs, score is' % (ei + 1)), score

    if verbose: sys.stdout.write('done.\n')
