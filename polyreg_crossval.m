
function [trainErrors, testErrors, bestD] = polyreg_crossval(x, y, maxDegree)
    % Splitting data into two halves randomly into training and testing
    % part
    N = length(x);
    rand = randperm(N);  
    trainIdx = rand(1:round(N/2)); 
    testIdx = rand(round(N/2)+1:end);
    
    xTrain = x(trainIdx);
    yTrain = y(trainIdx);
    xTest = x(testIdx);
    yTest = y(testIdx);
    
    trainErrors = zeros(maxDegree, 1);
    testErrors = zeros(maxDegree, 1);
    D=1;
    while D<=maxDegree
        % we train the polynomial model with degree D
        [trainErr, ~, testErr] = polyregg(xTrain, yTrain, D, xTest, yTest);
        
        trainErrors(D) = trainErr;
        testErrors(D) = testErr;
        D=D+1;
    end
    
    % Finding the degree with minimum testing error
    [minTestError, bestD] = min(testErrors);
    fprintf('The best degree is D = %d with a minimum testing error of %.4f\n', bestD, minTestError);
   
    % Plotting  training and testing errors
    figure;
    plot(1:maxDegree, trainErrors, 'r-', 'LineWidth', 2); hold on;
    plot(1:maxDegree, testErrors, 'b-', 'LineWidth', 2);
    xlabel('Polynomial Degree');
    ylabel('Error');
    legend('Training Error', 'Testing Error');
    title('Training and Testing Errors');
    
    
    plot(bestD, testErrors(bestD), 'bo', 'MarkerSize', 10, 'LineWidth', 2);
    text(bestD, testErrors(bestD), sprintf('Best D = %d', bestD), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    
    % Plotting the best-fitting polynomial model
    figure;
    [~, bestModel] = polyregg(xTrain, yTrain, bestD);
    q = linspace(min(x), max(x), 1000);
    qq = zeros(length(q), bestD);
    for i = 1:bestD
        qq(:,i) = q.^(bestD-i);
    end
    plot(x, y, 'X'); hold on;
    plot(q, qq * bestModel, 'r-', 'LineWidth', 2);
    xlabel('x');
    ylabel('y');
    title(sprintf('Best Polynomial Degree(%d)', bestD));
end
