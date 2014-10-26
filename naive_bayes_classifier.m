% Train a Naive Bayes Classifier to classifiy speculative language

%% Get unique set of words

% Shuffle sentences
n_labeled_sentences = size(labeled_sentences, 1);
labeled_sentences = labeled_sentences(randperm(n_labeled_sentences), :);

% Get training set
train_size = floor(size(labeled_sentences, 1) / 2);  % Size of training set
Xtrain = labeled_sentences(1:train_size, 1);
ytrain = cell2mat(labeled_sentences(1:train_size, 2));

% Remove commas, periods, brackets () from sentences in training set
% Remove trailing spaces (otherwise strsplit will return
for si = 1: length(Xtrain)
    Xtrain{si} = regexprep(Xtrain{si}, '[,\.\(\)]', '');
    Xtrain{si} = strtrim(Xtrain{si});
end

% Get unique words (we call this the dictionary)
sentences_concatenated = strjoin(Xtrain');  % Adds a space delimiter
all_words = strsplit(sentences_concatenated);
dictionary = unique(all_words);

%% Calculate P(speculative | word)

n_unique_words = length(dictionary);

% No. of speculative sentences that a word appears in
cell_of_zeros = num2cell(zeros(1, n_unique_words));
spec_counts = containers.Map(dictionary, cell_of_zeros);

% No. of unspeculative sentences that a word appears in
unspec_counts = containers.Map(dictionary, cell_of_zeros);

fprintf('No. of unique words: %i\n', n_unique_words);
fprintf('No. of sentences in training set: %i\n', length(Xtrain));

%% Populate marginal probabilities with a hash table

for si = 1:length(Xtrain)

    % Get unique words in this sentence
    sentence = Xtrain{si};
    words = unique(strsplit(sentence));

    for wi = 1:length(words)
        word = words{wi};
        if ytrain(si)
            spec_counts(word) = spec_counts(word) + 1;
        else
            unspec_counts(word) = unspec_counts(word) + 1;
        end
    end
    
    if mod(si, 1000) == 1
        fprintf('%5i ', si);
    end
end

% Marginal probabilities, P(word | speculative) & P(word | unspeculative)
spec_marginals = containers.Map(dictionary, cell_of_zeros);
unspec_marginals = containers.Map(dictionary, cell_of_zeros);

% Normalize marginal probabilities
n_speculative = sum(ytrain);
n_unspeculative = length(Xtrain) - n_speculative;
for wi = 1:length(dictionary)
    word = dictionary{wi};
    
    % P(word | spec) = #(spec sentences with word) / #(spec sentences)
    spec_marginals(word) = spec_counts(word) / n_speculative;
    
    % P(word | unspec) = #(unspec sentences with word) / #(unspec sentences)
    unspec_marginals(word) = unspec_counts(word) / n_unspeculative;
end
