clc
clear

rootdir = 'minimum_images/';

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

net = alexnet;
inputSize = net.Layers(1).InputSize;


augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainImages);
augimdsTest = augmentedImageDatastore(inputSize(1:2),testImages);
augimdsValidate = augmentedImageDatastore(inputSize(1:2),validateImages);

layer = 'fc8';
xTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
xValidate = activations(net,augimdsValidate, layer,'OutputAs','rows');
xTest = activations(net,augimdsTest,layer,'OutputAs','rows');

yTrain = trainImages.Labels;
yTest = testImages.Labels;
yValidate = validateImages.Labels;

% Distinct leaf subset
save('minimum_leaf_features.mat', 'xTrain', 'yTrain', "yValidate", "xValidate", "yTest", "xTest")