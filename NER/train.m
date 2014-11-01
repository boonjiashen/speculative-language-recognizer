%% Neural network for named entity recognition
%
%%======================================================================
%% STEP 0: Declare parameters

windowSize = 5;  % number of words per window
n = 50;  % length of one word vector
inputSize = windowSize * n;   % number of input units 
hiddenSize = 25;     % number of hidden units 
outputSize = 1;   % number of output units

lambda = 0.001;     % weight decay parameter   
eta = 0.0001;  % learning rate of gradient descent

train_filename = 'data/train';  % file containing labeled training data
test_filename = 'data/dev';  % file containing labeled test data

%%======================================================================
%% STEP 1.1: Load word matrix

% Creates a hashtable word2vec, containing |V| keys, that maps a
% word to its feature column vector
load_word_matrix;

vocab = word2vec.keys()';  % column cell array of words in word matrix
unknown_word = 'UUUNKKK';  % special word that replaces words that are in
                           % training data but not in vocab
start_token = '<s>';  % token to represent start of sentence
end_token = '</s>';  % token to represent end of sentence
                           
%% STEP 1.2: Load training and test data

% Load training/test data which are annotated sentences. Each word in
% every sentence is labeled either a PERSON or not.
% Resulting data are mx2 cell arrays that contains labeled words. The
% first column is a series of sentences broken down into one
% word/punctuation per cell. The second column is 1 if the word/punctuation
% is a person, else it's 0.
train_data = load_annotated_sentences(train_filename);
test_data = load_annotated_sentences(test_filename);

% Change words in training data to lower case since the words for the
% initial feature vectors are also in lower case
train_data(:, 1) = lower(train_data(:, 1));
test_data(:, 1) = lower(test_data(:, 1));

% Replace words that aren't in the vocab with the special unknown word
train_data(~ismember(train_data(:, 1), vocab), 1) = {unknown_word};
test_data(~ismember(test_data(:, 1), vocab), 1) = {unknown_word};

%% STEP 1.3: Obtain random parameters theta

Theta1 = randInitializeWeights(inputSize, hiddenSize);
Theta2 = randInitializeWeights(hiddenSize, outputSize);
theta = [Theta1(:); Theta2(:)];

%%======================================================================
%% STEP 2: Implement nnCostFunction

% grab some words as a column of cells
some_words = vocab(randi(length(vocab), [windowSize 1]));

some_vectors = word2vec.values(some_words);  % grab word vector columns
x = cell2mat(some_vectors);  % concatenate the vectors and make into a row
y = 1;  % dummy value
[cost, grad] = nnCostFunction([theta; x], inputSize, hiddenSize, outputSize, y, lambda);

%%======================================================================
%% STEP 3: Gradient Checking

if false
    % First, lets make sure your numerical gradient computation is correct for a
    % simple function.  After you have implemented computeNumericalGradient.m,
    % run the following: 
    checkNumericalGradient();

    % Now we can use it to check your cost function and derivative calculations
    % for the sparse autoencoder.  
    fun = @(x) nnCostFunction(x, inputSize, hiddenSize, outputSize, y, lambda);
    numgrad = computeNumericalGradient(fun, [theta; x]);

    % Use this to visually compare the gradients side by side
    disp([numgrad grad]); 

    % Compare numerically computed gradients with the ones obtained from backpropagation
    diff = norm(numgrad-grad)/norm(numgrad+grad);
    disp(diff); % Should be small. In our implementation, these values are
                % usually less than 1e-9.

                % When you got this working, Congratulations!!! 
end

%%======================================================================
%% STEP 4: Train neural network by stochastic gradient descent

% Run gradient descent over training examples
context_size = (windowSize - 1) / 2;  % no. of words to pad at start
costs = zeros(1, size(train_data, 1));
for ei = context_size + 1: size(train_data, 1) - context_size
% for ei = 5:13
    
    % Grab words in window
    words = train_data(ei - context_size: ei + context_size);
    
    % Grab label of the center word
    y = train_data{ei, 2};

    % Signify the start and end of the sentence (if any) with appropriate
    % tokens
    words = replace_sentence_start_and_end(words, start_token, end_token);
    
    % Grab word vectors to make an input vector
    x = cell2mat(word2vec.values(words'));
        
    % Calculate error derivatives w.r.t weights and input vector
    [cost, grad] = nnCostFunction([theta; x], ...
        inputSize, hiddenSize, outputSize, y, lambda);
    
    % Update weights (this is everything in the gradient vector except the
    % inputSize no. of elements at the end)
    dtheta = -eta * grad(1:end - inputSize);
    theta = theta + dtheta;
    
    % Update input vectors
    dx = grad(end - inputSize + 1: end);
    for wi = 1: windowSize
        word = words{wi};
        
        % Change in word vector for current word
        dx_i = -eta * dx((wi - 1) * n + 1: wi * n);
        
        % Update word vector in hashtable. This means that if one word
        % appears multiple times in the window, it'll be updated more than
        % once.
        word2vec(word) = word2vec(word) + dx_i;
    end

    % Remember cost at this iteration for graph plots
    costs(ei) = cost;
    
    if mod(ei, 10000) == 0
        fprintf('Done with example %i\n', ei);
    end
end

%% Plot change in error (a.k.a cost)

step_size = 3000;  % no. of iterations that we summarize into a point

% Chop off the tail of the matrix so that we can average more easily
costs_cropped = costs(1: floor(length(costs) / step_size) * step_size);

% Average cost for each step size
ave_costs = mean(reshape(costs_cropped, step_size, []));
 
%% Test neural network on test data

% Get weights between input and hidden layer
Theta1 = reshape(theta(1:hiddenSize * (inputSize + 1)), ...
                 hiddenSize, (inputSize + 1));

% Get weights between hidden and output layer
start_ind = 1 + (hiddenSize * (inputSize + 1));
end_ind = start_ind + outputSize * (hiddenSize + 1) - 1;
Theta2 = reshape(theta(start_ind: end_ind), ...
                 outputSize, (hiddenSize + 1));

% Range of indices of interest, in test data. We chop off a bit of the head
% and tail of the data so as not to worry about boundary conditions
ind_range = context_size + 1: size(test_data, 1) - context_size;

% Run test data through neural network and compare prediction with true
% label!
y_test = cell2mat(test_data(ind_range, 2));
predictions = zeros(size(y_test));
for ei = ind_range
% for ei = 5:13
    
    % Grab words in window
    words = test_data(ei - context_size: ei + context_size);
    
    % Grab label of the center word
    y = test_data{ei, 2};

    % Signify the start and end of the sentence (if any) with appropriate
    % tokens
    words = replace_sentence_start_and_end(words, start_token, end_token);
    
    % Grab word vectors to make an input vector
    x = cell2mat(word2vec.values(words'));
    
    % Make prediction
    confidence = sigmoid(Theta2 * [1; tanh(Theta1 * [1; x])]);
    prediction = confidence > 0.5;
    
    % Remember prediction
    predictions(ei - ind_range(1) + 1) = prediction;
end

%% Calculate performance metrics on test set

% Tally number of true/false positives/negatives
tp = sum(predictions(y_test == 1) == 1);
tn = sum(predictions(y_test == 0) == 0);
fp = sum(predictions(y_test == 0) == 1);
fn = sum(predictions(y_test == 1) == 0);

precision = tp / (tp + fn);
recall = tp / (tp + fp);
F1 = 2 * (precision * recall) / (precision + recall);

test_accuracy = sum(predictions == y_test) / length(predictions);
fprintf('Test accuracy = %.1f%%\n', test_accuracy * 100);
fprintf('Precision = %.2f | recall = %.2f | F1 = %.2f\n', precision, recall, F1);