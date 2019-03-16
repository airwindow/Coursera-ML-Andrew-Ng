function y_matrix = rollYVector(y_vector, num_labels)
%UNROLLYVECTOR Summary of this function goes here
%   Detailed explanation goes here
    % passed-in y_vector is supposed to be 1 * m vector (where m is the size of sample)
    m = size(y_vector, 1);
    y_matrix = zeros(m, num_labels);
    for i = 1 : m
        y_matrix(i, y_vector(i)) = 1;
    end
end

