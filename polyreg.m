function [err, model, errT] = polyreg(x, y, D, xT, yT, lambda)
%
% Performs multivariate regularized regression using polynomial features
%
%    function [err, model, errT] = polyreg(x, y, D, xT, yT, lambda)
%
% x = matrix of input features for training (NxM where N = # of samples, M = # of features)
% y = vector of output scalars for training
% D = order of the polynomial being fit (set to 1 for linear regression)
% xT = matrix of input features for testing
% yT = vector of output scalars for testing
% lambda = regularization parameter
% err = average squared loss on training
% model = vector of polynomial parameter coefficients
% errT = average squared loss on testing

[N, M] = size(x);  % N = number of samples, M = number of features

% Create the design matrix with polynomial features
xx = zeros(N, D * M);
for i = 1:D
    xx(:, (i-1)*M + 1:i*M) = x.^i;
end

% Regularization term (L2 regularization)
I = eye(size(xx, 2));  % Identity matrix for regularization
model = pinv(xx' * xx + lambda * I) * (xx' * y);  % Regularized model

%  training error
err = (1/(2 * N)) * sum((y - xx * model).^2) + (lambda/(2 * N)) * sum(model(2:end).^2);

% Testing error
if nargin >= 5 
    [NT, ~] = size(xT);
    xxT = zeros(NT, D * M);
    for i = 1:D
        xxT(:, (i-1)*M + 1:i*M) = xT.^i;
    end
    errT = (1/(2 * NT)) * sum((yT - xxT * model).^2);
else
    errT = NaN;
end

end


