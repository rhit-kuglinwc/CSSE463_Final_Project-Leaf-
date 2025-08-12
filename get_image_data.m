rootdir = [char(pwd) char('\archive (4)\Plants_2\')];
subdir = [rootdir 'train'];

trainImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

% Make datastores for the validation and testing sets similarly.
subdir = [rootdir 'test'];
testImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

subdir = [rootdir 'validate'];
validateImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

% REPLACE WITH THE NET THAT YOU ARE USING
net = xception;

% Anonymous functions for matlab would be like a lambda in python
extractCode = @(s) regexp(s, '\((P\d+)', 'tokens', 'once');

% can apply the function to the whole cell array with cell fun
% need to use cell str to make the categorical it cell array
% then need to change back to categorical
yTrain = categorical(cellfun(@(s) extractCode(s), cellstr(trainImages.Labels)));
yTest = categorical(cellfun(@(s) extractCode(s), cellstr(testImages.Labels)));
yValidate = categorical(cellfun(@(s) extractCode(s), cellstr(validateImages.Labels)));


trainImages.Labels = yTrain;
validateImages.Labels = yValidate;
testImages.Labels = yTest;

inputSize = net.Layers(1).InputSize;

augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainImages);
augimdsTest = augmentedImageDatastore(inputSize(1:2),testImages);
augimdsValidate = augmentedImageDatastore(inputSize(1:2),validateImages);

numClasses = numel(categories(yTrain));

save('leaf_features.mat', 'augimdsTrain', 'augimdsTest', "augimdsValidate", "numClasses")