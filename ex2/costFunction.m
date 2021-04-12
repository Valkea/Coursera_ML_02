function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

	%% Cost function J(Θ)

	% Method 1 : iterations
        %%for i = 1:m
	%%	thetaSum = 0;
	%%	for j = 1:size(theta,1)
	%%		thetaSum += theta(j)*X(i,j);
	%%	end
	%%	sigmo = sigmoid(thetaSum);
        %%        J += -y(i) * log(sigmo) - (1-y(i)) * log(1-sigmo);
        %%end
        %%J *= 1/m;

	% Method 2 : vectorized
	sigmo = sigmoid(X*theta);
	J = -(1/m) * ( log(sigmo)'*y + log(1-sigmo)'*(1-y) ); % log or log10 ?
	
	%% Gradient Descent

	% Method 1 : iterations
	%%for i = 1:m
	%%	thetaSum = 0;
	%%	for j = 1:size(theta,1)
	%%		thetaSum += theta(j)*X(i,j);
	%%	end

	%%	sigmo = sigmoid(thetaSum);

	%%	for k = 1:size(theta,1)
	%%		grad(k) += (sigmo - y(i)) * X(i,k);	
	%%	end
	%%end

	%%grad *= 1/m

	% Method 2 : vectorized
	sigmo = sigmoid(X*theta);
	grad =  1/m * X' * (sigmo-y);

% =============================================================

end
