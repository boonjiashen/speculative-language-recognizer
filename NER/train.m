%% Neural network for named entity recognition
%
%%======================================================================
%% STEP 0: Declare parameters

windowSize = 3;  % number of words per window
inputSize = windowSize * 50;   % number of input units 
hiddenSize = 25;     % number of hidden units 
outputSize = 1;   % number of output units

lambda = 0;     % weight decay parameter   

%%======================================================================
%% STEP 1: Load vocabulary and initialize word vectors

% Creates a |V|x1 cell array vocab and a hashtable word2vec that maps a
% word to its feature column vector
load_data;

%  Obtain random parameters theta
Theta1 = randInitializeWeights(inputSize, hiddenSize);
Theta2 = randInitializeWeights(hiddenSize, outputSize);
theta = [Theta1(:); Theta2(:)];

%%======================================================================
%% STEP 2: Implement nnCostFunction

some_words = vocab([1, 4, 7]);  % grab some words as a column of cells
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


% %%======================================================================
% %% STEP 5: Visualization 
% 
% W1 = reshape(opttheta(1:hiddenSize*inputSize), hiddenSize, inputSize);
% display_network(W1', 12); 
% 
% print -djpeg weights.jpg   % save the visualization to a file 
% 
% 
