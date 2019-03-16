function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

% caculate the Cost at given theta
h = X * theta;
% caculate the cost comes from square loss
J_normal =  1 / (2 * m) * ((h - y)' * (h - y));
% caculate the cost comes from regularization
J_reg = lambda / (2 * m) * (theta(2 : end)' * theta(2 : end));
J = J_normal + J_reg;

% calculate the graident from square loss portion
grad_normal = 1 / m .* (X' * (h - y));
% calculate the graident from regularization portion
grad_reg = (lambda / m) .* theta;
grad_reg(1) = 0;
grad = grad_normal + grad_reg;
% =========================================================================

grad = grad(:);

end
