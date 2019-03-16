function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% prepare candiates lsit for C and sigma (increase by 10 each step with proper anchors)
C_candidates = [0.01, 0.03, 0.07, 0.1, 0.3, 0.7, 1, 3, 7, 10, 30, 70, 100];
sigma_candidates = [0.01, 0.03, 0.07, 0.1, 0.3, 0.7, 1, 3, 7, 10, 30, 70, 100];
% prepare the errors matrix to record val_error for each [C, sigma]
val_errors = zeros(length(C_candidates), length(sigma_candidates));

for i = 1 : length(C_candidates)
    for j = 1 : length(sigma_candidates)
        C = C_candidates(i);
        sigma = sigma_candidates(j);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        % record the val_error
        val_errors(i, j) = mean(double(predictions ~= yval));
    end
end

% find the indexes of min_error and retrieve related C and sigma
min_error = min(val_errors(:));
% there are may be multiple [C, sigma] has the same min_val_error
% only need to get the first one
[min_i, min_j] = find(val_errors == min_error, 1);
C = C_candidates(min_i);
sigma = sigma_candidates(min_j);
% =========================================================================

end
