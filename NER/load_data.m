% Loads the vocabulary and each word's initial fixed length feature vector

% file of feature vectors, one row per feature vector
vector_filename = 'data/wordVectors.txt';

% file containing vocabulary, one line per word
words_filename = 'data/vocab.txt';

%% Read in vocabulary

% Read in vocab into a cell array, each row as one word
text_block = fileread(words_filename);
vocab = strsplit(text_block, '\n')';

% Read in word vectors, each column is the feature vector of a word
word_vectors = importdata(vector_filename)';