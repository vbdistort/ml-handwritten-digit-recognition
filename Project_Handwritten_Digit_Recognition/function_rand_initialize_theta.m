function [ theta1, theta2 ] = function_rand_initialize_theta( input_layer_size, hidden_layer_size, output_layer_size )
%FUNCTION_INITIALIZE_THETA Summary of this function goes here
%   Detailed explanation goes here

% random numbers between -2 and +2
% rand(m,n) returns m x n matrix of random value in interval (0,1)
%theta1 = (2 - 4*rand(input_layer_size + 1, hidden_layer_size)); 
%theta2 = (2 - 4*rand(hidden_layer_size + 1, output_layer_size));

epsilon_init = 0.12;
theta1 = rand(input_layer_size + 1, hidden_layer_size) * 2 * epsilon_init - epsilon_init;
theta2 = rand(hidden_layer_size + 1, output_layer_size) * 2 * epsilon_init - epsilon_init;

end

