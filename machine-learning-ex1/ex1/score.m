function [pred] = score(theta, mu, sigma, x)
%SCORE Summary of this function goes here
%   Detailed explanation goes here
%     col_cnt = length(sigma)
%     x_vector = zeros(col_cnt)
%     for i = 1 : col_cnt
%         x_vector(i) = (x(i) - mu(i)) / sigma(i)
%     end
    x_vector = (x - mu) ./ sigma
    % add the column for intercept
    x_vector = [1, x_vector]
    pred = x_vector * theta
end

