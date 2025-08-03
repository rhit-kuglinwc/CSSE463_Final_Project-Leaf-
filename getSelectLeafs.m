function [xOut, yOut] = getSelectLeafs(X, Y, classes)
    index = find(ismember(Y, classes));
    yOut = Y(index);
    xOut = X(index, :);
end