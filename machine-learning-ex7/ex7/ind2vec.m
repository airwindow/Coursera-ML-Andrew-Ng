function encoded_matrix = ind2vec(idx, K)
% covert a vector of idx into a one-hot encoded matrix
% assume m = size(idx, 1), idx[i] belongs to [1 : K]
% if idx(i) = k, then encoded_matrix(i, k) = 1, all other encoded_matrix(i,
% j) = 0.
m = size(idx, 1);
encoded_matrix = zeros(m, K);
% map each idx(i) into on-hot encoded row
for i = 1 : m
   encoded_matrix(i, idx(i)) = 1; 
end

end