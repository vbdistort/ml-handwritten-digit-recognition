function [ h, display_array , matrixY] = function_display_data(X, vectorY, figureTitle)
%FUCNTION_DISPLAY_DATA Summary of this function goes here
%   This function displays 2D data stored in X in a grid

[m n] = size(X); % returns number of rows and columns in X -> 100 x 784

image_width = round(sqrt(n)); % in our case it is 28 since every image is
image_height = n/image_width; % 28x28 matrix

figure('name',figureTitle);
% Gray Image
colormap(gray);

display_rows = floor(sqrt(m));
display_cols = ceil(m/display_rows);

pad = 1;

display_array = - ones(pad + display_rows * (image_height + pad), ...
                       pad + display_cols * (image_width + pad));

curr_ex = 1;
for j = 1 : display_rows
    for i = 1 : display_cols
        if curr_ex > m
            break;
        end
        
        max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (image_height + pad) + (1:image_height), ...
		              pad + (i - 1) * (image_width + pad) + (1:image_width)) = ...
						reshape(X(curr_ex, :), image_height, image_width) / max_val;
		curr_ex = curr_ex + 1;
    end
end

matrixY = reshape(vectorY(1 : display_rows*display_cols), display_rows, display_cols);
% Display Image
h = imagesc(display_array', [-1 1]);

% Do not show axis
axis image off

drawnow;

end