"""Utility functions useful for word2vec and doc2vec applications
"""

import argparse
from bs4 import BeautifulSoup


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


def download_XML_from_Biomed(src_filename="cc13902.xml", dst_filename="dummy"):
    """Returns a XML textblock from a Biomed article

    Resource:
    http://www.biomedcentral.com/about/datamining
    """

    from ftplib import FTP
    ftp = FTP('ftp.biomedcentral.com')  # connect to server
    ftp.login('datamining', '$8Xguppy')  # login with username and password
    ftp.cwd('articles')  # walk to the directory with all the XML files

    if False:

        # Assign XML file to string
        data = []
        def handle_binary(more_data):
            """Tells ftp to append downloaded text to a list.

            Source:
            http://stackoverflow.com/questions/18772703/read-a-file-in-buffer-from-ftp-python
            """

            data.append(more_data)

        ftp.retrbinary('RETR %s' % src_filename, callback=handle_binary)

    else:

        # Write XML file to disk
        with open(dst_filename, 'w') as fid:
            ftp.retrbinary('RETR %s' % src_filename, callback=fid.write)

    ftp.close()
