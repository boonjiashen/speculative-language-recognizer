function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

% Assert that theta is either a column vector or a scalar
assert(size(theta, 2) == 1);

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

epsilon = 10^-4;

for ei = 1:length(theta)

    % Create sparse vector with zeros everywhere except ei where it's 1
    sparse_vector = zeros(size(theta));
    sparse_vector(ei) = 1;
    
    % Calculate difference between J(theta+) and J(theta-)
    delta_J = J(theta + epsilon * sparse_vector) - J(theta - epsilon * sparse_vector);

    % Get error gradient w.r.t one element of theta
    numgrad(ei) = delta_J / (2 * epsilon);
    
end






%% ---------------------------------------------------------------
end
