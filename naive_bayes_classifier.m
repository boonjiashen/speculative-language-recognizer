% Train a Naive Bayes Classifier to classifiy speculative language

%% Get unique set of words

% Get sentences and labels in their own variables
all_sentences = labeled_sentences(:, 1);
are_speculative = cell2mat(labeled_sentences(:, 2));

% Remove commas, periods, brackets () from sentences
% Remove trailing spaces (otherwise strsplit will return
for si = 1: length(all_sentences)
    all_sentences{si} = regexprep(all_sentences{si}, '[,\.\(\)]', '');
    all_sentences{si} = strtrim(all_sentences{si});
end

% Get unique words (we call this the dictionary)
sentences_concatenated = strjoin(all_sentences');  % Adds a space delimiter
all_words = strsplit(sentences_concatenated);
dictionary = unique(all_words);

%% Calculate P(speculative | word)

n_unique_words = length(dictionary);

% No. of sentences that a word appears in
sentence_counts = zeros(n_unique_words, 1);

% No. of speculative sentences that a word appears in
spec_counts = zeros(n_unique_words, 1);

fprintf('No. of unique words: %i\n', n_unique_words);
fprintf('No. of sentences: %i\n', length(all_sentences));

%% Populate marginal probabilities with a hash table
sentence_counts = containers.Map(dictionary, num2cell(zeros(1, n_unique_words)));
spec_counts = containers.Map(dictionary, num2cell(zeros(1, n_unique_words)));

for si = 1:length(all_sentences)

    % Get unique words in this sentence
    sentence = all_sentences{si};
    words = unique(strsplit(sentence));

    for wi = 1:length(words)
        word = words{wi};
        sentence_counts(word) = sentence_counts(word) + 1;

        if are_speculative(si)
            spec_counts(word) = spec_counts(word) + 1;
        end
    end
    
    if mod(si, 1000) == 1
        fprintf('%5i ', si);
    end
end
%% Populate marginal probablilities by iterating over sentences

if false
    for si = 1:length(all_sentences)

        % Get unique words in this sentence
        sentence = all_sentences{si};
        words = unique(strsplit(sentence));

        % Get the index of these words in the dictionary
        mask = ismember(dictionary, words);

        % Update marginal probabilities
        sentence_counts(mask) = sentence_counts(mask) + 1;
        if are_speculative(si)
            spec_counts(mask) = spec_counts(mask) + 1;
        end

        if mod(si, 100) == 1
            fprintf('%5i ', si);
        end
    end
end

%% Populating marginal probabilities by iterating over dictionary

if false
    for wi = 1: n_unique_words

        word = unique_words{wi};
        occurrence_inds = strfind(labeled_sentences(:, 1), word);
        does_appear = ~cellfun(@isempty, occurrence_inds);

        % No. of sentences that a word appears in
        sentence_count = sum(does_appear);

        % No. of speculative sentences that a word appears in
        spec_count = sum(does_appear & are_speculative);

        % Update marginal probabilities
        sentence_counts(wi) = sentence_count;
        spec_counts(wi) = spec_count;

        if mod(wi, 100) == 1
            fprintf('%i ', wi);
        end
    end
end
