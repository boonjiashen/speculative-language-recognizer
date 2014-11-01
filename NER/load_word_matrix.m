%% Loads the vocabulary and each word's initial fixed length feature vector

% file of feature vectors, one row per feature vector
vector_filename = 'data/wordVectors.txt';

% file containing vocabulary, one line per word
words_filename = 'data/vocab.txt';

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