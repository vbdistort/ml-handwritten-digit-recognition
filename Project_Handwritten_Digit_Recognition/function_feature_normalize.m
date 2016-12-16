function [X_norm, mu, sigma] = function_feature_normalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% Written below in comments is previous normalising method written by me
% Updating it!!

% mu = mean(X);
% sigma = std(X);
% X_temp = zeros(size(X));
% 
% for i = 1 : size(X,2)
%     if sigma(:,i) == 0
%         X_temp(:,i)=0;
%     else
%         X_temp(:,i)=(X(:,i)-mu(:,i))/sigma(:,i);
%     end;
% end;

% X_norm=X_temp;
 
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = nanstd(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

X_norm(isnan(X_norm)) = 0;

% ============================================================

end
