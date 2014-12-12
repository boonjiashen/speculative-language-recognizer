"""Extracts and prints labeled phrases from BioScope XML files

Each line printed is in the format: INT WORD1 WORD2 WORD3...
where INT is 0 for a non-speculative phrase and 1 for a speculative phrase.
WORD1 WORD2 etc are the corresponding phrase that has been tokenized and
separated by spaces. These tokens may be words or punctuation.

Troubleshooting
--------------------
If you're piping the output of this file into another file, e.g.
>> python extract_labeled_phrases.py data.xml > data.txt
you may encounter a UnicodeEncodeError because the default encoding of Python
2.7 is not set. In that case, try the following
>> PYTHONIOENCODING=UTF-8 python extract_labeled_phrases.py data.xml > data.txt
Resource:
http://stackoverflow.com/questions/4545661/unicodedecodeerror-when-redirecting-to-file
"""

import nltk
import argparse
from bs4 import BeautifulSoup


def get_speculative_scopes(tag):
    """Returns text of all speculative scopes in a BeautifulSoup tag

    If a tag is:
    SOMEBODY
      <scope>
        <spec_cue>MAY</spec_cue>
        EAT
        <scope>
          <spec_cue>HYPOTHETICAL-SOUNDING</spec_cue>
          FOOD
        </scope>
      </scope>
    LATER  
    Then this function returns a list of two strings:
    "MAY EAT HYPOTHETICAL-SOUNDING FOOD" and
    "HYPOTHETICAL-SOUNDING FOOD"

    """

    texts = [subtag.get_text()
        for subtag in tag.find_all('xcope')
        if subtag.find('cue', type='speculation')
        ]

    return texts
    

if __name__ == "__main__":


    ######################## Parse command-line arguments ##################### 

    parser = argparse.ArgumentParser()

    # Add required argument of XML file
    parser.add_argument('filename', metavar='filepath', type=str,
                               help='XML file to extract phrases from')

    # Optionally generate multiple instances per speculative sentence by
    # recursion
    msg = """
    Do not optionally generate multiple instances per speculative sentence by recursion
    (recursion is on by default)"""
    parser.add_argument('--norecurse', action='store_true', help=msg)

    # Grab arguments from stdin
    args = parser.parse_args()

    # Filename of BioScope XML file
    filename = args.filename
    norecurse = args.norecurse


    ######################### Load file, parse file ########################### 

    # Load filename into a beautiful soup
    fid = open(filename)
    textblock = fid.read()
    fid.close()
    soup = BeautifulSoup(textblock)

    # Gather negative (non-speculative) and positive (speculative) phrases
    neg_phrases, pos_phrases = [], []
    for tag in soup.find_all('sentence'):

        # Push full phrase into appropriate labeled set
        contains_spec_cue = tag.find('cue', type='speculation')
        if contains_spec_cue:
            pos_phrases.append(tag.get_text())
        else:
            neg_phrases.append(tag.get_text())

        # Recursively generate more speculative instances from speculative
        # sentences (if allowed by user)
        if contains_spec_cue and not norecurse:
            pos_phrases.extend(get_speculative_scopes(tag))


    ######################### Print labeled phrases ###########################

    for label, phrases in [(0, neg_phrases), (1, pos_phrases)]:

        for phrase in phrases:

            words = [word.lower() for word in nltk.word_tokenize(phrase)]
            print label, ' '.join(words)
