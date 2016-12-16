function [ J, grad ] = function_cost_lr(theta, X, Y, lambda)
%FUNCTION_COST_LR Summary of this function goes here
%   Detailed explanation goes here

[m n] = size(X);

J = 0;
grad = zeros(n,1);
% 
% prediction = function_sigmoid(X*theta);
% cost_each = Y.*log(prediction) + (1-Y).*log(1-prediction);
% 
% theta_temp = theta(2:end);
% 
% J = -(1/m)*(sum(cost_each)) + (lambda/(2*m))*(theta_temp*theta_temp');
% 
% grad(1) = (1/m)*( (X(:,1))'*(prediction-Y) );
% 
% grad(2:end) = (1/m)*( (X(:,2:end))'*(prediction-Y) ) + (lambda/m)*theta(2:end);

y = Y;

% Compute cost function
pred = X*theta;

templog(:,1) = log(function_sigmoid(pred));
templog(:,2) = log(1-(function_sigmoid(pred)));
tempy(:,1) = y;
tempy(:,2) = 1-y;
temp1 = (templog(:,1))'*tempy(:,1);
temp2 = (templog(:,2))'*tempy(:,2);

% Formula for cost function.
J = (1/m)*(-temp1-temp2) + (lambda/(2*m))*(theta'*theta);

grad(1,1) = (1/m)*((function_sigmoid(pred)-y)'*X(:,1)); 

grad(2:end,1) = (1/m)*((function_sigmoid(pred)-y)'*X(:,2:end))' + (lambda/m)*theta(2:end);


end

