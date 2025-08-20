clc
% clear

load("leaf_features.mat")

kernelFunctions = {'linear', 'rbf', 'polynomial'};
BoxConstraints = [0.01, 0.1, 1, 10, 100];
kernelScale = [1, 5, 10, 15, 20];

results = cell(size(kernelFunctions, 2) * ...
    size(BoxConstraints, 2) * ...
    size(kernelScale, 2), 4);

i = 1;

for kfunc = kernelFunctions
    for bconstraint = BoxConstraints
        for kscale = kernelScale
            template = templateSVM('KernelFunction', kfunc{1}, ...
                'BoxConstraint', bconstraint, ...
                'Standardize', true, ...
                'KernelScale', kscale);
    
            net = fitcecoc(xTrain, yTrain, 'Learners', template);
            predictions = predict(net, xValidate);
    
            % Evaluate the model's performance
            accuracy = sum(predictions == yValidate) / numel(yValidate);
            results(i, :) = {kfunc{1}, bconstraint, kscale, accuracy};
            fprintf('Kernel: %s, Box Constraint: %.2f, Kernel Scale %d, Accuracy: %.2f\n', ...
                results{i, :});            
            i = i + 1;
        end
    end
end

[~, idx] = max(cell2mat(results(:, 4)));

fprintf('BEST -> Kernel: %s, Box Constraint: %.2f, Kernel Scale %d, Accuracy: %.2f\n', ...
                results{idx, :});




