function [ all_theta ] = function_one_vs_all(X, Y, num_classes, lambda)
%FUNCTION_ONE_VS_ALL Summary of this function goes here
%   Detailed explanation goes here

X = [ones(size(X,1),1) X]; % +1 for theta0
[m n] = size(X);

% fprintf('\nFeature normalising training data for better results..');
% [X mu sigma] = function_feature_normalize(X);

   % intialize theta
   all_theta = zeros(num_classes, n);
   
   % use advanced optimisation algorithm fmincg to get local optima and
   % find optimal theta's for every class, that is, 0-9 digits
   
   
   for i = 1 : num_classes
       fprintf('\nUsing optimisation algorithm fmincg for class %d ......\n',i);
       initial_theta = zeros(n,1);
       options = optimset('GradObj','on','MaxIter',40);
       [optimal_theta] = fmincg( @(t)(function_cost_lr(t,X,(Y == i),lambda)), initial_theta, options );
       all_theta(i,:) = optimal_theta';
   end
   
end

