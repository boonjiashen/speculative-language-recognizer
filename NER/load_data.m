% Loads the vocabulary and each word's initial fixed length feature vector
% Also loads training data in the form of annotated sentences. Each word in
% every sentence is labeled either a PERSON or not.

% file of feature vectors, one row per feature vector
vector_filename = 'data/wordVectors.txt';

% file containing vocabulary, one line per word
words_filename = 'data/vocab.txt';

% file containing training data
train_filename = 'data/train';

%% Read in vocabulary and their feature vectors

% Read in vocab into a cell array, each row as one word
text_block = fileread(words_filename);
vocab = strsplit(strtrim(text_block), '\n')';

% Read in word vectors, each column is the feature vector of a word (hence
% the tranpose)
% This part will take the most computational time since it's a huge file.
word_vectors = importdata(vector_filename)';

% Create hash table that maps a word to its feature (column) vector
word2vec = containers.Map();
for wi = 1:length(vocab)
    word = vocab{wi};
    vector = word_vectors(:, wi);
    word2vec(word) = vector;
end

%% Read in training data

% Load training data as strings
fid = fopen(train_filename);
trainData = textscan(fid, '%s %s\n');  % first cell is words, second cell is labels
trainData = horzcat(trainData{:});  % contacenate to mx2 cell array
fclose(fid);

% Convert labels from strings '0'/'PERSON' to integers 0/1
for wi = 1: size(trainData, 1)
    label = trainData{wi, 2};
    isPerson = strcmp(label, 'PERSON') == 1;
    if isPerson
        trainData{wi, 2} = 1;
    else
        trainData{wi, 2} = 0;
    end
end