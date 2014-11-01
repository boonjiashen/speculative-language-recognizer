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

%%======================================================================
%% STEP 1: Load vocabulary and initialize word vectors

% Creates a hashtable word2vec, containing |V| keys, that maps a
% word to its feature column vector
% Also creates a mx2 cell array trainData that contains labeled words. The
% first column is a series of sentences broken down into one
% word/punctuation per cell. The second column is 1 if the word/punctuation
% is a person, else it's 0.
load_data;

vocab = word2vec.keys()';  % column cell array of words in word matrix
unknown_word = 'UUUNKKK';  % special word that replaces words that are in
                           % training data but not in vocab

% Change words in training data to lower case since the words for the
% initial feature vectors are also in lower case
trainData(:, 1) = lower(trainData(:, 1));

% Replace words that aren't in the vocab with the special unknown word
trainData(~ismember(trainData(:, 1), vocab), 1) = {unknown_word};

%  Obtain random parameters theta
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
%
% Hint: If you are debugging your code, performing gradient checking on smaller models 
% and smaller training sets (e.g., using only 10 training examples and 1-2 hidden 
% units) may speed things up.

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

%%======================================================================
%% STEP 4: Train neural network by stochastic gradient descent

% Run gradient descent over training examples
context_size = (windowSize - 1) / 2;  % no. of words to pad at start
costs = zeros(1, size(trainData, 1));
for ei = context_size + 1: size(trainData, 1) - context_size
% for ei = 5:13
    
    % Grab words in window
    words = trainData(ei - context_size: ei + context_size);
    
    % Grab label of the center word
    y = trainData{ei, 2};

    % Signify the start and end of the sentence (if any) with appropriate
    % tokens
    words = replace_sentence_start_and_end(words, '<s>', '</s>');
    
    % Grab word vectors to make an input vector
    x = zeros(inputSize, 1);
    for wi = 1: windowSize
        word = words{wi};
        vector = word2vec(word);
        x((wi-1) * n + 1: wi * n) = vector;
    end
        
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
    
    if mod(ei, 10000) == 1
        fprintf('Done with example %i\n', ei);
    end
end

%% Plot change in error (a.k.a cost)

step_size = 3000;  % no. of iterations that we summarize into a point

% Chop off the tail of the matrix so that we can average more easily
costs_cropped = costs(1: floor(length(costs) / step_size) * step_size);

% Average cost for each step size
ave_costs = mean(reshape(costs_cropped, step_size, []));
 
