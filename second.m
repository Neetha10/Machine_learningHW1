% Loading the dataset
load('dataset.mat'); 
n = size(X, 1);
Y(Y == 0) = -1; 
Y(Y == 1) = 1;  
%  (Z-score normalization)
mu = mean(X);
sigma = std(X);
X_normalized = (X - mu) ./ sigma;

% Split the dataset
idx_train = randsample(n, n / 2);
idx_test = setdiff(1:n, idx_train);
trnX = X_normalized(idx_train, :);
tstX = X_normalized(idx_test, :);
trnY = Y(idx_train, :);
tstY = Y(idx_test, :);

function error_rate = svc_error(trnX, trnY, tstX, tstY, kernel, alpha, bias, sigma)
    if strcmp(kernel, 'rbf')
        predictedY = svcoutput(trnX, trnY, tstX, 'rbf', alpha, bias, sigma);
    else
        predictedY = svcoutput(trnX, trnY, tstX, 'poly', alpha, bias);
    end
    error_rate = sum(predictedY ~= tstY) / length(tstY);
end


d_max = 5; % Max polynomial degree
log_costs = -10:1:10;
costs = 10.^log_costs; 
errors_poly = zeros(d_max, length(costs));

% SVM with Polynomial Kernels
for d = 1:d_max
    for ci = 1:length(costs)
        C = costs(ci); 
     
        [nsv, alpha, bias] = svc(trnX, trnY, 'poly', C, d); 
        
     
        errors_poly(d, ci) = svc_error(trnX, trnY, tstX, tstY, 'poly', alpha, bias);
    end
end

% Plot for Polynomial Kernel
[DegreeGrid, CostGrid] = meshgrid(1:d_max, 1:length(costs));
figure;
bar3(errors_poly');
title('Testing errors using polynomial kernel with different orders and costs');
xlabel('Polynomial Order');
ylabel('Cost (log scale)');
zlabel('Error');
set(gca, 'YTick', 1:length(costs), 'YTickLabel', log_costs);
set(gca, 'YScale', 'log');
print('-depsc', 'polynomial_errors.eps');

%% RBF Kernel Testing

% Parameters for RBF Kernel
log_sigmas = -10:1:10; 
sigmas = 10.^log_sigmas; 
errors_rbf = zeros(length(sigmas), length(costs));
% SVM with RBF Kernel
for si = 1:length(sigmas)
    sigma = sigmas(si); 
    for ci = 1:length(costs)
        C = costs(ci); 
        [nsv, alpha, bias] = svc(trnX, trnY, 'rbf', C, sigma);
        
        errors_rbf(si, ci) = svc_error(trnX, trnY, tstX, tstY, 'rbf', alpha, bias, sigma);
    end
end
% Plotting for RBF Kernel
[SigmaGrid, CostGrid] = meshgrid(1:length(sigmas), 1:length(costs));
figure;
bar3(errors_rbf');
title('Testing errors using RBF kernel with different sigmas and costs');
xlabel('Sigma (log scale)');
ylabel('Cost (log scale)');
zlabel('Error');
set(gca, 'YTick', 1:length(costs), 'YTickLabel', log_costs);
set(gca, 'XTick', 1:length(sigmas), 'XTickLabel', log_sigmas);
set(gca, 'YScale', 'log');
print('-depsc', 'rbf_errors.eps');
