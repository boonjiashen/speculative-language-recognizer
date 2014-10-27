function [ pos_loglikes, neg_loglikes ] = calculate_word_likelihoods( sentences, labels, smooth_term)
%CALCULATE_WORD_LIKELIHOODS Return P(word|positive) & P(word|negative)
%   sentences - a Nx1 cell array of sentences
%   labels - a Nx1 matrix where labels(i) is 1 if sentences{i} is a
%   positive sentence, and 0 otherwise.
%   pos_loglikes - a hashtable that maps a word to its log10 likelihood given a
%   positive class, i.e. P(word | positive)
%   neg_loglikes - same as pos_loglikes, except for the negative class
%   The two output arguments do not have to be of the same length.

%% Pre-process sentences

for si = 1: length(sentences)
    sentences{si} = strtrim(sentences{si});
end

% Get unique words (we call this the dictionary)
sentences_concatenated = strjoin(sentences');  % Adds a space delimiter
all_words = strsplit(sentences_concatenated);
dictionary = unique(all_words);
n_unique_words = length(dictionary);

% No. of positive sentences that a word appears in
cell_of_zeros = num2cell(zeros(1, n_unique_words));
pos_counts = containers.Map(dictionary, cell_of_zeros);

% No. of negative sentences that a word appears in
neg_counts = containers.Map(dictionary, cell_of_zeros);

%% Populate likelihoods with a hash table

for si = 1:length(sentences)

    % Get unique words in this sentence
    sentence = sentences{si};
    words = unique(strsplit(sentence));

    for wi = 1:length(words)
        word = words{wi};
        if labels(si)
            pos_counts(word) = pos_counts(word) + 1;
        else
            neg_counts(word) = neg_counts(word) + 1;
        end
    end
    
end

% Likelihoods, P(word | pos) & P(word | neg)
pos_loglikes = containers.Map(dictionary, cell_of_zeros);
neg_loglikes = containers.Map(dictionary, cell_of_zeros);

% Normalize likelihoods
% We take the log10 of these probabilities so they don't go too close to
% zero.
% We also perform Laplacian smoothing by hallucinating 1 example from each
% class.
n_pos = sum(labels);
n_neg = sum(~labels);
for wi = 1:length(dictionary)
    word = dictionary{wi};
    
    % P(word | pos) = #(pos sentences with word) / #(pos sentences)
    pos_loglikes(word) = log10((pos_counts(word)+smooth_term) / ...
        (n_pos + n_unique_words*smooth_term));
    
    % P(word | neg) = #(neg sentences with word) / #(neg sentences)
    neg_loglikes(word) = log10((neg_counts(word)+smooth_term) / ...
        (n_neg + n_unique_words*smooth_term));
end

end

