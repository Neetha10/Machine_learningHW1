load('problem1.mat');  % Load the dataset with x and y
maxDegree = 200;
[trainErrors, testErrors, bestD] = polyreg_crossval(x, y, maxDegree);
