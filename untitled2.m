% Load the dataset
load('problem2.mat');  % Assuming x and y are stored in this .mat file

% Define lambda values to test
lambda_values = 0:50:1000;

% Perform two-fold cross-validation
twoFoldCrossValidation(x, y, lambda_values);
