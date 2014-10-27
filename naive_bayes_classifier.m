% Train a Naive Bayes Classifier to classify speculative language

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
fprintf('done in %f sec.\n', toc);

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

%% Calculate marginal probability of words

fprintf('Calculating marginal probabilities of words... '); tic;
smooth_term = 1;
[pos_loglikes, neg_loglikes] = calculate_word_likelihoods( ...
        Xtrain, ytrain, smooth_term);
fprintf('done in %f sec.\n', toc);

%% Test classifier

fprintf('Testing %i examples... ', test_size); tic;
predictions = zeros(test_size, 1);
dictionary = pos_loglikes.keys;
n_sentences_with_new_words = 0;
for si = 1:test_size
    
    % Get unique words in this sentence
    sentence = Xtest{si};
    words = unique(strsplit(sentence));
    
    % Only look for words that are in the dictionary
    words_in_dictionary = words(ismember(words, dictionary));
    
    has_new_words = length(words) ~= length(words_in_dictionary);
    n_sentences_with_new_words = n_sentences_with_new_words + has_new_words;

    % Get non-normalize posterior probability for speculative language
    curr_pos_loglikes = pos_loglikes.values(words_in_dictionary);
    nonnorm_pos = sum(cell2mat(curr_pos_loglikes));
    
    % Get non-normalize posterior probability for unspeculative language
    curr_neg_loglikes = neg_loglikes.values(words_in_dictionary);
    nonnorm_neg = sum(cell2mat(curr_neg_loglikes));
    
    % The predicted class is the argmax of the non-normalized probabilities
    % over the possible classes (i.e. speculative or non-speculative)
    prediction = nonnorm_pos > nonnorm_neg;

    predictions(si) = prediction;

end
fprintf('done in %f sec.\n', toc);

accuracy = sum(predictions == ytest) / length(predictions);
fprintf('Accuracy = %f\n', accuracy);