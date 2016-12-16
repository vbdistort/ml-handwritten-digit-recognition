function [ J , grad ] = function_cost_nn( theta_vec, input_layer_size, hidden_layer_size ...
                                          , output_layer_size, num_classes, X, Y, lambda)
%FUNCTION_COST_NN Summary of this function goes here
%   Detailed explanation goes here

% theta1 785 x 40 , theta2 41 x 10
theta1 = reshape(theta_vec(1: (input_layer_size+1)*hidden_layer_size), input_layer_size + 1, hidden_layer_size);
theta2 = reshape(theta_vec((input_layer_size+1)*hidden_layer_size + 1: end), ...
                            hidden_layer_size + 1, output_layer_size);

m = size(X,1); % 10000
J = 0;
theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));

% COST (J) part

a1 = [ones(m,1) X]; % 10000 x 785

a2 = function_sigmoid(a1 * theta1); % 10000 x 40
a2 = [ones(m,1) a2]; % 10000 x 41

a3 = function_sigmoid(a2 * theta2); % 10000 x 10

vectorY = repmat([1:num_classes], m, 1) == repmat(Y, 1, num_classes); % 10000 x 10

theta1_no_bias = theta1(:, 2:end);
theta2_no_bias = theta2(:, 2:end);

J = (-1/m)*sum(sum(vectorY.*log(a3)+(1-vectorY).*log(1-a3))) + ...
    (lambda/(2*m))*(sum(sum(theta1_no_bias.^2)) + sum(sum(theta2_no_bias.^2)));


% GARDIENT (grad) part

Delta1 = zeros(size(theta1));
Delta2 = zeros(size(theta2));

for i = 1 : m

    a1_transpose = (a1(i,:))'; % 785 x 1

    a2_transpose = (a2(i,:))'; % 41 x 1

    a3_transpose = (a3(i,:))'; %10 x 1

    y_transpose = (vectorY(i,:))'; %10 x 1

    delta3 = a3_transpose - y_transpose; % 10 x 1
    delta2 = (theta2*delta3).*(a2_transpose.*(1-a2_transpose)); % 41 x 1
    % delta1 not required

    Delta2 = Delta2 + a2_transpose*delta3';  % 41 x 10
    Delta1 = Delta1 + a1_transpose*delta2(2:end)'; % 785 x 40
    
end

theta1_zero_bias = [ zeros(size(theta1, 1), 1) theta1_no_bias ];
theta2_zero_bias = [ zeros(size(theta2, 1), 1) theta2_no_bias ];

theta2_grad = (1/m)*Delta2 + (lambda/m)*(theta2_zero_bias);
theta1_grad = (1/m)*Delta1 + (lambda/m)*(theta1_zero_bias);


grad = [theta1_grad(:) ; theta2_grad(:)];

end

