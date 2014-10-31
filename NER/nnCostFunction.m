function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification, for a single training
%example
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. In
%   particular, the parameters include the training example since we
%   backpropagate the error all the way back to the input layer.
%   x is a input_layer_size-length column vector
%   y is a scalar.
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%   The sequence of parameters is [b1(:) W1(:) b2(:) W2(:) x], where b is
%   the weights of the bias units, W is the weights of the non-bias units
%   and x is the training example.
%

%% Reshape nn_params back into the parameters Theta1, Theta2 and x
% Theta are the weight matrices
% for our 2 layer neural network and x is the input vector

% First column of Theta1 is the bias weights, the other columns are the
% non-bias weights. Same goes for Theta2.

% Get weights between input and hidden layer
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

% Get weights between hidden and output layer
start_ind = 1 + (hidden_layer_size * (input_layer_size + 1));
end_ind = start_ind + num_labels * (hidden_layer_size + 1) - 1;
Theta2 = reshape(nn_params(start_ind: end_ind), ...
                 num_labels, (hidden_layer_size + 1));

x = nn_params(end - input_layer_size + 1: end);

assert(num_labels == 1);  % right now only works with 1 output unit

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
x_grad = zeros(size(x));

%% Forward propagation

% Notation: s_l is the no. of units (excluding bias) in the l-layer
a1 = x; % activation of input layer - size s_1 x 1
z2 = Theta1 * [1; a1]; % input to hidden layer - size s_2 x 1
a2 = sigmoid(z2); % activation of hidden layer - size s_2 x 1
z3 = Theta2 * [1; a2]; % input to output layer - size s_3 x 1
a3 = sigmoid(z3); % activation of output layer - size s_3 x 1

%% Compute cost
% This converts y to 1-hot encoding
y_1hot = y;

% Get binary cross entropy cost (non-regularized)
J = y_1hot .* -log(a3) + (1-y_1hot) .* -(log(1-a3));

% Get cost (regularized)
sumTheta1 = sum(sum(Theta1(:,2:end).^2));
sumTheta2 = sum(sum(Theta2(:,2:end).^2));
J = J + lambda / 2 * (sumTheta1 + sumTheta2);

%% Compute error derivatives w.r.t weights and input vector

% Get gradients by interating over examples
delta3 = a3 - y_1hot;
Theta2_grad = delta3 * [1; a2]'; % size of Theta2
delta2 = (Theta2(:,2:end)'*delta3) .* sigmoidGradient(z2);
Theta1_grad = delta2 * [1; a1]'; % size of Theta1
x_grad = Theta1(:, 2:end)' * delta2;

% Add the regularization term
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda*Theta2(:,2:end);

%% Unroll gradients

grad = [Theta1_grad(:); Theta2_grad(:); x_grad];


end
