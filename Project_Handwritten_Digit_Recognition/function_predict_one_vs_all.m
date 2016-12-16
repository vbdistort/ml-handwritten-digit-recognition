function [ result, accuracy ] = function_predict_one_vs_all( theta, X, Y, test_size )
%FUNCTION_PREDICT_ONE_VS_ALL Summary of this function goes here
%   Detailed explanation goes here
m = size(X,1);
X = [ones(m,1) X];
% fprintf('\nFeature normalising test data for better results..');
% [X mu sigma] = function_feature_normalize(X);

prediction = function_sigmoid(X*theta');

[max_value max_index] = max(prediction');

result = max_index';

%temp = reshape(result(:), sqrt(dummy_test_size), sqrt(dummy_test_size));
% result_reshape = reshape(result(:),20,20);
% size(result)
% size(Y)
accuracy = sum(sum((result == Y),1))/size(Y,1)

end

