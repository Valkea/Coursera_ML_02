function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

        %% Cost function J(Θ)

        % Method 1 : iterations
        %%for i = 1:m
        %%      thetaSum = 0;
        %%      for j = 1:size(theta,1)
        %%              thetaSum += theta(j)*X(i,j);
        %%      end
        %%      sigmo = sigmoid(thetaSum);
        %%        J += -y(i) * log(sigmo) - (1-y(i)) * log(1-sigmo);
        %%end
        %%J *= 1/m;

        % Method 2 : vectorized
        %sigmo = sigmoid(X*theta);
        %J = -(1/m) * ( log(sigmo)'*y + log(1-sigmo)'*(1-y) ); 

	%% Cost Regularization

	%thetaSumSqr = 0;
        %for j = 2:size(theta,1)
        %	thetaSumSqr += theta(j).^2;
        %end
	%J += (lambda/(2*m)) * thetaSumSqr;
        
	% Method 3 : Vectorized with Regularization inside
        sigmo = sigmoid(X*theta);
        J = -(1/m) * ( log(sigmo)'*y + log(1-sigmo)'*(1-y) ) + (lambda/(2*m)) * sum(theta(2:end).^2);

        %% Gradient Descent 
 
        % Method 1 : iterations 
        %%for i = 1:m 
        %%      thetaSum = 0; 
        %%      for j = 1:size(theta,1) 
        %%              thetaSum += theta(j)*X(i,j); 
        %%      end 
 
        %%      sigmo = sigmoid(thetaSum); 
 
        %%      for k = 1:size(theta,1) 
        %%              grad(k) += (sigmo - y(i)) * X(i,k);      
        %%      end
        %%end 
        %%grad *= 1/m;
	
	
	% Method 2 : vectorized
        %sigmo = sigmoid(X*theta);
        %grad =  1/m * X' * (sigmo-y);
	%
	%% Gradient Regularization
 
        %for j = 2:size(theta,1)
	%	grad(j) += lambda/m * theta(j);
	%end
        

	% Method 3 : Vectorized with Regularization inside

        sigmo = sigmoid(X*theta);
        grad =  1/m * X' * (sigmo-y) + ( lambda/m * theta .* [0, ones(1, length(theta)-1)]' );


% =============================================================

end
