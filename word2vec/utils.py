import argparse


def get_parser(default_n_epochs=5, default_min_word_count=5):
    """Returns an argument parser intended for scripts utilizing word2vec and
    doc2vec
    
    Has arguments - debug, verbose, min_count, n_epochs
    """

    parser = argparse.ArgumentParser()

    # Add argument for debug statements
    parser.add_argument("--debug",
            help="print debug messages", action="store_true")

    # Add argument for number of training epochs
    parser.add_argument("--n_epochs", type=int,
            help="number of training epochs (default=%i)" % default_n_epochs,
            default=default_n_epochs
            )

    # Add argument for more verbose stdout
    parser.add_argument("-v", "--verbose",
            help="print status during program execution", action="store_true")

    # Min count to allow a word in the vocabulary
    parser.add_argument('--min_count', type=int,
            help='min count to allow a word in the vocabulary (default=' +
            str(default_min_word_count) + ')',
            default=default_min_word_count,
            )

    return parser


