% Train a Naive Bayes Classifier to classifiy speculative language

%% Pre-process sentences
% Remove commas, periods, brackets () from all sentences
% Remove trailing spaces (otherwise strsplit will return

fprintf('Pre-processing sentences... '); tic;
for si = 1: size(labeled_sentences, 1)
    sentence = labeled_sentences{si, 1};
    sentence = regexprep(sentence, '[,\.\(\)]', '');
    sentence = strtrim(sentence);
    labeled_sentences{si, 1} = sentence;
end
fprintf('done.\n'); toc;

%% Partition dataset to training and test set

% Shuffle sentences
n_labeled_sentences = size(labeled_sentences, 1);
labeled_sentences = labeled_sentences(randperm(n_labeled_sentences), :);

% Partition out training set from dataset
train_size = floor(0.5 * size(labeled_sentences, 1));  % Size of training set
Xtrain = labeled_sentences(1:train_size, 1);
ytrain = cell2mat(labeled_sentences(1:train_size, 2));

% Partition out test set from dataset
test_size = floor(0.3 * size(labeled_sentences, 1));  % Size of test set
test_size = min(500, test_size);
Xtest = labeled_sentences(train_size+1:train_size+test_size, 1);
ytest = cell2mat(labeled_sentences(train_size+1:train_size+test_size, 2));

%% Get unique set of words

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

fprintf('Calculating marginal probabilities of words... '); tic;
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
    
end

% Marginal probabilities, P(word | speculative) & P(word | unspeculative)
spec_logmarginals = containers.Map(dictionary, cell_of_zeros);
unspec_logmarginals = containers.Map(dictionary, cell_of_zeros);

% Normalize marginal probabilities
% We take the log10 of these probabilities so they don't go too close to
% zero.
% We also perform Laplacian smoothing by hallucinating 1 example from each
% class.
n_speculative = sum(ytrain);
n_unspeculative = length(Xtrain) - n_speculative;
for wi = 1:length(dictionary)
    word = dictionary{wi};
    
    % P(word | spec) = #(spec sentences with word) / #(spec sentences)
    spec_logmarginals(word) = log10((spec_counts(word)+1) / (n_speculative+2));
    
    % P(word | unspec) = #(unspec sentences with word) / #(unspec sentences)
    unspec_logmarginals(word) = log10((unspec_counts(word)+1) / (n_unspeculative+2));
end

fprintf('done.\n'); toc;

%% Test classifier

fprintf('Testing %i examples... ', test_size); tic;
predictions = zeros(test_size, 1);
for si = 1:test_size
    
    % Get unique words in this sentence
    sentence = Xtest{si};
    words = unique(strsplit(sentence));
    
    % Only look for words that are in the dictionary
    words_in_dictionary = words(ismember(words, dictionary));
    
    % Get non-normalize posterior probability for speculative language
    nonnorm_spec = sum(cell2mat(spec_logmarginals.values(words_in_dictionary)));
    
    % Get non-normalize posterior probability for unspeculative language
    nonnorm_unspec = sum(cell2mat(unspec_logmarginals.values(words_in_dictionary)));
    
    % The predicted class is the argmax of the non-normalized probabilities
    % over the possible classes (i.e. speculative or non-speculative)
    prediction = nonnorm_spec > nonnorm_unspec;
    
    predictions(si) = prediction;

end
fprintf('done.\n'); toc;