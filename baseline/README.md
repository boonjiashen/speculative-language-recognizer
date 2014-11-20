Instructions to compare baseline algorithms
------------------------------------------------------------
    >> # Extract labeled phrases from BioScope XML files by running
    >> python ../word2vec/extract_labeled_phrases.py data.xml > data.txt
    >> # Run the comparison script with this text file as input
    >> python compare_baselines.py data.txt


Troubleshooting
------------------------------------------------------------
When piping the labeled phrases to a file, e.g.
    >> python extract_labeled_phrases.py data.xml > data.txt
you may encounter a UnicodeEncodeError because the default encoding of Python
2.7 is not set. In that case, try the following
    >> PYTHONIOENCODING=UTF-8 python extract_labeled_phrases.py data.xml > data.txt
Resource:
[http://stackoverflow.com/questions/4545661/unicodedecodeerror-when-redirecting-to-file][]
