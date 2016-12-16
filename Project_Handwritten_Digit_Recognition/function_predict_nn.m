function [ result, accuracy ] = function_predict_nn( X, Y, lambda, theta_vec , num_classes, ...
                               input_layer_size, hidden_layer_size, output_layer_size, dummy_test_size )
%FUNCTION_PREDICT_NN Summary of this function goes here
%   Detailed explanation goes here

    m = size(X,1);
%      fprintf('\nFeature normalising test data for better results..');
%      [X mu sigma] = function_feature_normalize(X);

    theta1 = reshape(theta_vec(1: (input_layer_size+1)*hidden_layer_size), ...
                        input_layer_size + 1, hidden_layer_size);
    theta2 = reshape(theta_vec((input_layer_size+1)*hidden_layer_size + 1: end), ...
                            hidden_layer_size + 1, output_layer_size);
                        
    a1 = [ones(m,1) X]; % 10000 x 785

    a2 = function_sigmoid(a1 * theta1); % 10000 x 40
    a2 = [ones(m,1) a2]; % 10000 x 41

    a3 = function_sigmoid(a2 * theta2); % 10000 x 10

    %vectorY = repmat([1:num_classes], m, 1) == repmat(Y, 1, num_classes); % 10000 x 10
    
    
    [max_value max_index] = max(a3');

    result = max_index';
    
    %temp = reshape(result(:), sqrt(dummy_test_size), sqrt(dummy_test_size));
    
    accuracy = sum(sum((result == Y),1))/m
end

