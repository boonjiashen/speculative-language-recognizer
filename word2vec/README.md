Train a speculative language recognizer using architecture introduced in Le &
Mikolov '14 (Distributed Representations of Sentences and Documents)


Train and test recognizer
------------------------------
    >> # Extract labeled phrases from BioScope XML files by running
    >> python ../word2vec/extract_labeled_phrases.py abstracts.xml > phrases.txt
    >> # Optionally extract more phrases
    >> python ../word2vec/extract_labeled_phrases.py full_papers.xml >> phrases.txt
    >> # Run the comparison script with this text file as input
    >> python recognizer_with_preprocessed_BioScope.py phrases.txt


Known bugs
------------------------------
* `utils.get_Biomed_XML_as_string()` reads some characters as question marks.
  These characters include &#945; (alpha) and $#946; (beta).
* `utils.write_Biomed_XML_to_file()` only partially downloads XML files
