Train a speculative language recognizer using architecture introduced in Le &
Mikolov '14 (Distributed Representations of Sentences and Documents)

#Train recognizer on labeled and unlabeled data
---

1. Generate file of labeled phrases (see below)
2. (Optionally) Download some articles from [Biomed Central](http://www.biomedcentral.com/about/datamining) into a folder. This folder should contain nothing but Biomed articles. This is your unlabeled data.
3. Run script
    
<b></b>

    >> python recognizer_with_Biomed_and_BioScope.py <filename_of_labeled_data> <Biomed_folder> <no. of Biomed articles to train on>
    >> # For example:
    >> python recognizer_with_Biomed_and_BioScope.py labeled_phrases.txt data/articles/ 5


If you have no Biomed articles, set the last two arguments as an existing folder and zero articles.

#Generate file of labeled phrases from BioScope XML files
---
    >> # Extract labeled phrases from BioScope XML files by running
    >> python ../word2vec/extract_labeled_phrases.py abstracts.xml > phrases.txt
    >> # Optionally extract more phrases
    >> python ../word2vec/extract_labeled_phrases.py full_papers.xml >> phrases.txt


#Known bugs
---
* `utils.get_Biomed_XML_as_string()` reads some characters as question marks.
  These characters include &#945; (alpha) and &#946; (beta).
* `utils.write_Biomed_XML_to_file()` only partially downloads XML files
* The following commands give different results even though they shouldn't

<b></b>

    >> run recognizer_with_Biomed_and_BioScope.py ../data/abstracts_and_full_papers_norecurse.txt ../../articles 0 --verbose
    >> run recognizer_with_Biomed_and_BioScope.py ../data/abstracts_and_full_papers_norecurse.txt data 0 --verbose
