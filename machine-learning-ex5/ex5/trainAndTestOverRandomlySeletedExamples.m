function [train_error, val_error] = ... 
    trainAndTestOverRandomlySeletedExamples(X_poly, y, X_poly_val, yval, lambda, sampling_size, sampling_cnt)

m = size(y, 1);
train_errors = zeros(sampling_cnt, 1);
val_errors = zeros(sampling_cnt, 1);
% set seed for random generator
rand_seed = RandStream('shr3cong');

for i = 1 : sampling_cnt
    % randomly select sampling_cnt indexes(without replcement) from [1 :m]
    select_idxes = randsample(rand_seed, m, sampling_size, false);
    % based on the selected indexes to get corresponding train & val data
    % set
    selected_X_train = X_poly(select_idxes, :);
    selected_y_train = y(select_idxes, :);
    selected_Xval = X_poly_val(select_idxes, :);
    selected_yval = yval(select_idxes, :);
    % train the model to get theta
    [theta] = trainLinearReg(selected_X_train, selected_y_train, lambda);
    % caculate the train & val error
    diff_train = selected_X_train * theta - selected_y_train;
    diff_val = selected_Xval * theta - selected_yval;
    train_errors(i) = train_errors(i) + (1 / (2 * sampling_size)) * (diff_train' * diff_train);
    val_errors(i) = val_errors(i) + (1 / (2 * size(selected_yval, 1))) * (diff_val' * diff_val);
end

train_error = mean(train_errors);
val_error = mean(val_errors);

end

