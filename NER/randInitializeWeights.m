function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%
%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(L_out+L_in+1);   % we'll choose weights uniformly from the interval [-r, r]
W_no_bias = rand(L_out, L_in) * 2 * r - r;  % weight excluding bias
b = zeros(L_out, 1);  % bias

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
W = [b W_no_bias];

end

