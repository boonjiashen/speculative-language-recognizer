function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   x, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification, for a single training
%example
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%   x is a input_layer_size-length column vector
%   y is a scalar.
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%   The sequence of parameters is [b1(:) W1(:) b2(:) W2(:) x]
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

assert(num_labels == 1);  % right now only works with 1 output unit

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% Notation: s_l is the no. of units (excluding bias) in the l-layer
a1 = x'; % activation of input layer - size s_1 x 1
z2 = Theta1 * [1; a1]; % input to hidden layer - size s_2 x 1
a2 = sigmoid(z2); % activation of hidden layer - size s_2 x 1
z3 = Theta2 * [1; a2]; % input to output layer - size s_3 x 1
a3 = sigmoid(z3); % activation of output layer - size s_3 x 1

% This converts y to 1-hot encoding
y_1hot = y;

% Get binary cross entropy cost (non-regularized)
J = y_1hot .* -log(a3) + (1-y_1hot) .* -(log(1-a3));

% Get cost (regularized)
sumTheta1 = sum(sum(Theta1(:,2:end).^2));
sumTheta2 = sum(sum(Theta2(:,2:end).^2));
J = J + lambda / 2 * (sumTheta1 + sumTheta2);

% Get gradients by interating over examples
delta3 = a3 - y_1hot;
Theta2_grad = delta3 * [1; a2]'; % size of Theta2
delta2 = (Theta2(:,2:end)'*delta3) .* sigmoidGradient(z2);
Theta1_grad = delta2 * [1; a1]'; % size of Theta1

% Add the regularization term
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda*Theta2(:,2:end);









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end