% Classification using substring matching
% This script extracts cue phrases indicating speculation from the 
% Bioscope corpus and classifies new text based on presence of cue phrases.
%
% Assumes that XML data has already been loaded
% and n_sentences, sentenceElements and labeledSentences are populated

% Partition out training set from dataset
idx = randperm(n_sentences);
train_size = floor(0.5 * size(labeled_sentences, 1));  % Size of training set
train_idx = idx(1:train_size);

% Build hashtable, each key of which is a unique cue
cues = containers.Map();

for i = 1:train_size
    sentenceElement = sentenceElements.item(train_idx(i)-1);

    cueElements = sentenceElement.getElementsByTagName('cue');
    for ci = 1:cueElements.getLength
        cueElement = cueElements.item(ci-1);
        if strcmp(cueElement.getAttribute('type'), 'speculation') == 1
            word = char(cueElement.getTextContent);
            cues(word) = 1;
        end
    end
end

% Regular expression corresponding to cues
cues_regexp = strjoin(cues.keys, '|');
cues_regexp = strcat('( |\.)(', cues_regexp);
cues_regexp = strcat(cues_regexp, ')( |\.)');

% Partition out test set from dataset
test_size = floor(0.3 * size(labeled_sentences, 1));  % Size of test set
test_size = min(500, test_size);
test_idx = idx(train_size+1:train_size+test_size);

% Classify test set
labels = cell2mat(labeled_sentences(test_idx, 2));
predicted = zeros(test_size, 1);

for i = 1:test_size
    sentence = cell2mat(labeled_sentences(test_idx(i), 1));
    match = regexp(sentence, cues_regexp);
    predicted(i) = size(match, 1) ~= 0;
end

% Compute accuracy
correct = sum(predicted == labels);
accuracy = correct/test_size;
fprintf('Accuracy = %f\n', accuracy);