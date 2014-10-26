% This script loads data from XML file that comes from the Bioscope corpus
%
% Dependencies: 1) an XML file from the corpus, 2) BioScope.dtd from the
% corpus. Both dependences should be in the same folder.
%
% Resource to load XML in MATLAB
% http://blogs.mathworks.com/community/2010/06/28/using-xml-in-matlab/

% File containing sentences and labels
filename = 'data/abstracts.xml';

xDoc = xmlread(filename);  % DOM specified by Java
sentenceElements = xDoc.getElementsByTagName('sentence');  % get sentences
n_sentences = sentenceElements.getLength;  % Number of sentences

fprintf('No. of sentences in database: %i\n', n_sentences);
n_sentences = n_sentences;
fprintf('Constrained to this no. of sentences: %i\n', n_sentences);

% mx2 cell containing sentences as MATLAB str in the first column and the
% true label in the second column (1 if sentence is speculative, 0 otherwise)
labeled_sentences = cell(n_sentences, 1);

% Parse sentence elements
for i=1:n_sentences
    sentenceElement = sentenceElements.item(i-1);
    string = char(sentenceElement.getTextContent);
    
    % An element with <cue type="speculation"> indicates a
    % speculative sentence
    isSpeculative = 0;
    cueElements = sentenceElement.getElementsByTagName('cue');
    for ci = 1:cueElements.getLength
        cueElement = cueElements.item(ci-1);
        if strcmp(cueElement.getAttribute('type'), 'speculation') == 1
            isSpeculative = 1;
        end
    end
    
    labeled_sentences{i, 1} = string;
    labeled_sentences{i, 2} = isSpeculative;
end
