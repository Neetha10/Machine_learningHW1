function [theta, errors, risks] = logistic_regression(X, Y, eta, tol, max_iter)
    % X: Input data matrix (N x d)
    % Y: Output labels (N x 1)
    % eta:  (step size)
    % tol: Tolerance for convergence
    % max_iter: Maximum number of iterations
    
    [N, d] = size(X); 
    X = [ones(N, 1), X];
    d = d + 1; 
    theta = zeros(d, 1); 
    errors = []; 
    risks = [];
    
    for iter = 1:max_iter
        % this is the prediction variable
        yi= 1 ./ (1 + exp(-X * theta));
        
        % gradient of cost function
        gradient = (1 / N) * X' * (yi - Y);
        
        % updating the theta
        theta = theta - eta * gradient;
        
        %  empirical risk with logistic loss
        emp_risk = (1 / N) * sum(-Y .* log(yi) - (1 - Y) .* log(1 - yi));
        risks = [risks; emp_risk];
        
        % Compute classification error
        predictions_binary = yi >= 0.5;
        classification_error = mean(predictions_binary ~= Y);
        errors = [errors; classification_error];
        
        % convergence check
        if norm(gradient) < tol
            break;
        end
    end
    
    % computing and plotting the decision boundary
    figure;
    % Plot the data
    gscatter(X(:,2), X(:,3), Y, 'rb', 'xo');
    hold on;
   
    x1 = linspace(min(X(:,2)), max(X(:,2)), 100);
    x2 = -(theta(1) + theta(2) * x1) / theta(3);
    
   
    plot(x1, x2, 'k-', 'LineWidth', 2);
    xlabel('Feature1');
    ylabel('Feature2');
    title('Decision Boundary');
    legend('Class 0', 'Class 1', 'Decision Boundary');
    hold off;
    
   
    figure;
    subplot(2, 1, 1);
    plot(errors, 'r-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Classification Error');
    title('Classification Error Over Iterations');
    
    subplot(2, 1, 2);
    plot(risks, 'b-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Empirical Risk');
    title('Empirical Risk Over Iterations');
end

load dataset4;

%  parameters
eta = 0.01; 
tol = 1e-6; 
max_iter = 1000; 

%calling the function
[theta, errors, risks] = logistic_regression(X, Y, eta, tol, max_iter);
