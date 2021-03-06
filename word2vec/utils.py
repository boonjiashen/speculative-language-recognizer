"""Utility functions useful for word2vec and doc2vec applications
"""

import argparse
import nltk
from ftplib import FTP
from bs4 import BeautifulSoup


def multigen(gen_func):
    """Decorator that makes a generator function reuseable.

    Add @multigen on top of the function header
    Resource:
    http://stackoverflow.com/questions/1376438/how-to-make-a-repeating-generator-in-python
    """

    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)
    return _multigen


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


def retrieve_sentences_from_Biomed_textblock(textblock,
        sentenize=nltk.data.load('tokenizers/punkt/english.pickle').tokenize):
    """Returns a list of sentences from a textblock of a single Biomed article

    sentenize is a function that takes a string and returns a list of
    sentences.

    Resource:
    http://www.biomedcentral.com/about/datamining
    """

    # If the textblock has '\n', BeautifulSoup cannot find <p> tags
    # Not sure why.
    textblock = textblock.replace('\n', '')

    soup = BeautifulSoup(textblock)
    sentences = [sentence
            for tag in soup.find_all('p')
            for sentence in sentenize(tag.get_text())]

    return sentences


def get_Biomed_FTP_object():
    """Returns a Python FTP object after logging in to Biomed Central

    Object has to be closed after use.

    >>> ftp = get_Biomed_FTP_object()  # login
    >>> ftp.cwd('articles')  # cd to another directory
    >>> # etc
    >>> ftp.close()
    """

    ftp = FTP('ftp.biomedcentral.com')  # connect to server
    ftp.login('datamining', '$8Xguppy')  # login with username and password
    ftp.cwd('articles')  # walk to the directory with all the XML files

    return ftp


def get_Biomed_XML_as_string(src_filename="cc13902.xml", ftp=None):
    """Downloads a Biomed article and returns it as a string.

    If ftp is None, program logs in to Biomed FTP server, downloads the file
    and logs out. Otherwise, uses the given ftp object and doesn't close ftp.
    """

    data = []
    def handle_binary(more_data):
        """Tells ftp to append downloaded text to a list.

        Source:
        http://stackoverflow.com/questions/18772703/read-a-file-in-buffer-from-ftp-python
        """

        data.append(more_data)

    # Open connection to FTP
    ftp_given = ftp is not None
    if not ftp_given:
        ftp = get_Biomed_FTP_object()

    # Download XML file
    ftp.retrbinary('RETR %s' % src_filename, callback=handle_binary)

    if not ftp_given:
        ftp.close()

    # Join the chunks we downloaded and return as one long string
    return ''.join(data)


def write_Biomed_XML_to_file(src_filename="cc13902.xml", dst_filename="dummy",
    ftp=None,):
    """Downloads a Biomed article and writes it to file.

    If ftp is None, program logs in to Biomed FTP server, downloads the file
    and logs out. Otherwise, uses the given ftp object and doesn't close ftp.

    Resource:
    http://www.biomedcentral.com/about/datamining
    """

    # Open connection to FTP
    ftp_given = ftp is not None
    if not ftp_given:
        ftp = get_Biomed_FTP_object()

    # Write XML file to disk
    with open(dst_filename, 'wb') as fid:
        ftp.retrbinary('RETR %s' % src_filename, callback=fid.write)

    if not ftp_given:
        ftp.close()


if __name__ == '__main__':


    # Read list of XML filenames
    filenames_file = 'data/Biomed_filenames'  # File storing filenames
    with open(filenames_file) as fid:
        xml_filenames = fid.read().splitlines()

    # Download a couple of Biomedical articles and grab their sentences
    n_filenames = 3  # No. of articles we want to download
    sentences = []  # Biomedical sentences from Biomed articles
    ftp = get_Biomed_FTP_object()  # Login to FTP and cd to appropriate dir
    for filename in xml_filenames[:n_filenames]:
        textblock = get_Biomed_XML_as_string(ftp=ftp, src_filename=filename)
        curr_sentences = retrieve_sentences_from_Biomed_textblock(textblock)

        sentences.extend(curr_sentences)

    ftp.close()  # Close FTP connection
