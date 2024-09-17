function twoFoldCrossValidation(x, y, lambda_values)
    % Splitting dataset into two folds as asked in question
    N = size(x, 1);
    fold1_x = x(1:N/2, :);
    fold1_y = y(1:N/2, :);
    fold2_x = x(N/2 + 1:end, :);
    fold2_y = y(N/2 + 1:end, :);
    
    % Store the errors
    trainError = zeros(length(lambda_values), 1);
    testError = zeros(length(lambda_values), 1);
    
  
    i=1;
    while i<=length(lambda_values)
        lambda = lambda_values(i);
        
        
        [trainerr1, ~, testerr1] = polyreg(fold1_x, fold1_y, 1, fold2_x, fold2_y, lambda);
        
        
        [trainerr2, ~, testerr2] = polyreg(fold2_x, fold2_y, 1, fold1_x, fold1_y, lambda);
        %first fold for traning and second fold for testinf , vice versa
        %performing avarage on traninig and testing errors
        trainError(i) = (trainerr1 + trainerr2) / 2;
        testError(i) = (testerr1 + testerr2) / 2;
        i=i+1;
    end
    
    % lambda with minimal testing error
    [~, minIdx] = min(testError);
    best_lambda = lambda_values(minIdx);
    
    % Plotting
    figure;
    plot(lambda_values, trainError, 'r', 'LineWidth', 2);
    hold on;
    plot(lambda_values, testError, 'b', 'LineWidth', 2);
    xlabel('Lambda');
    ylabel('Error');
    legend('Training Error', 'Testing Error');
    title('Training and Testing Error');
    
    % best_lamba
    plot(best_lambda, testError(minIdx), 'bo', 'MarkerSize', 10, 'LineWidth', 2);
    text(best_lambda, testError(minIdx), sprintf('\\lambda = %.2f', best_lambda), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end
