%% This script reduces the dimensionality of word vectors and plots words in a 2D space

% fprintf('Loading word matrices... '); tic;
% load_word_matrix;
% fprintf('done in %.1f sec\n', toc);

%% Normalize word vectors

% Create a condensed word matrix
vocab = word2vec.keys();
% vocab = vocab(randperm(length(vocab), 100));  % choose |V| words
% vocab = {'stand', 'standing', 'sit', 'sitting', 'run', 'running', 'kill', 'killing'};
word_matrix = cell2mat(word2vec.values(vocab));  % n x |V| where n is word vector length
[n, vocab_size] = size(word_matrix);

% Perform feature normalization over each feature
mius = mean(word_matrix, 2);  % mean of each feature
sigmas = std(word_matrix, 1, 2);  % stddev of each feature
X = zeros(size(word_matrix));  % normalized feature matrix
for i = 1:n
    miu = mius(i);
    sigma = sigmas(i);
    X(i, :) = (word_matrix(i, :) - miu) / sigma;
end

%% PCA

Sigma = 1 / vocab_size * (X * X');
[U, S, V] = svd(Sigma);  % U is the principle components, S is the diagonal matrix
K = 2;  % no. of dimensions to reduce to
X_reduced = zeros(K, vocab_size);  % X after dim-reduction
U_reduce = U(:, 1:K);  % principle components that we'll project on to
for ii = 1: vocab_size    
    x = X(:, ii);
    x_reduced = U_reduce' * x;
    X_reduced(:, ii) = x_reduced;
end

%% Plot dimensionally of reduced word vectors for some choice words

hold on;
choice_words = {'kill', 'killing', 'killed', 'eat', 'eating', 'ate', ...
    'paris', 'france', 'beijing', 'china', 'amsterdam', 'netherlands', 'rome', 'italy'};
X_choice = zeros(2, length(choice_words));
for ii = 1: length(choice_words)
    word = choice_words{ii};
    vind = find(strcmp(word, vocab));  % get index in vocab
    x = X_reduced(1, vind);
    y = X_reduced(2, vind);
    text(x, y, word);
    
    X_choice(:, ii) = [x; y];
end

% Scale axes by the min and max x/y coordinates since text() doesn't do it
% automatically
xymin = min(X_choice');
xymax = max(X_choice');
axis([xymin(1) xymax(1) xymin(2) xymax(2)]) 

axis equal