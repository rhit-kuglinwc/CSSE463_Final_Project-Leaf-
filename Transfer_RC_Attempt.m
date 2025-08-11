clear
clc
rootdir = 'Plants_2/';
subdir = [rootdir 'train'];

trainImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

subdir = [rootdir 'test'];

testImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

subdir = [rootdir 'valid'];

valImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

% [train_feats, test_feats, valid_feats] = CNNFeatureExtract(trainImages, testImages, valImages);


% REPLACE WITH THE NET THAT YOU ARE USING
net = imagePretrainedNetwork("resnet18",NumClasses=22);
inputSize = net.Layers(1).InputSize;


% Anonymous functions for matlab would be like a lambda in python
extractCode = @(s) regexp(s, '\((P\d+)', 'tokens', 'once');

% can apply the function to the whole cell array with cell fun
% need to use cell str to make the categorical it cell array
% then need to change back to categorical
% trainImages.Labels = categorical(cellfun(@(s) extractCode(s), cellstr(trainImages.Labels)));
% testImages.Labels = categorical(cellfun(@(s) extractCode(s), cellstr(testImages.Labels)));
% valImages.Labels = categorical(cellfun(@(s) extractCode(s), cellstr(valImages.Labels)));

augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainImages);
augimdsTest = augmentedImageDatastore(inputSize(1:2),testImages);
augimdsValid = augmentedImageDatastore(inputSize(1:2),valImages);





% net = imagePretrainedNetwork("resnet18",NumClasses=12);
% 
% inputSize = net.Layers(1).InputSize;

% analyzeNetwork(net)



% ends up being layer fc8
layer = networkHead(net);

net = freezeNetwork(net,LayerNamesToIgnore=layer);

% augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainImages);
% augimdsValid = augmentedImageDatastore(inputSize(1:2),valImages);

options = trainingOptions("adam", ...
    ValidationData=augimdsValid, ...
    MiniBatchSize=32, ...
    ValidationFrequency=5, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false);

net = trainnet(augimdsTrain,net,"crossentropy",options);


% The functions freezeNetwork, networkHead, and findSource come from:
% https://www.mathworks.com/help/deeplearning/ug/retrain-neural-network-to-classify-new-images.html
function net = freezeNetwork(net,args)
% netFrozen = freezeNetwork(net) sets the learning rate factors of all the
% learnable parameters of the specified network to zero.
%
% netFrozen = freezeNetwork(net,LayersToIgnore=layerClassNames) also
% specifies the layer types to leave the learning rate factors unchanged.

arguments
    net dlnetwork
    args.LayerNamesToIgnore = string.empty;
    args.LayerTypesToIgnore = string.empty;
end

layerNamesToIgnore = args.LayerNamesToIgnore;
layerTypesToIgnore = args.LayerTypesToIgnore;

% Find names of layers to freeze.
layerNames = {net.Layers.Name}';
layerClassNames = arrayfun(@class,net.Layers,UniformOutput=false);

idxLayersToFreeze = ...
    ~contains(layerClassNames,layerTypesToIgnore) ...
    & ~ismember(layerNames,layerNamesToIgnore);

layersToFreeze = {net.Layers(idxLayersToFreeze).Name}';

% Create table of layer and parameter name pairs.
idxParametersToFreeze = ismember(net.Learnables.Layer,layersToFreeze);
tbl = net.Learnables(idxParametersToFreeze,1:2);

% Loop over parameters to freeze.
for i = 1:size(tbl,1)
    layerName = tbl.Layer(i);
    parameterName = tbl.Parameter(i);
    net = setLearnRateFactor(net,layerName,parameterName,0);
end

end

function [layerName,learnableNames] = networkHead(net)
%NETWORKHEAD Network head learnables
%   layerName = networkHead(net) returns the name of the last
%   convolution or fully connected layer in the network NET.
%   [layerName,learnableNames] = networkHead(net) also returns
%   the paths of the learnable parameters.

outputName = net.OutputNames;

supportedLayers = [
    "FullyConnectedLayer" 
    "Convolution2DLayer"];

layerName = outputName;
while ~isempty(layerName)
    layer = getLayer(net,layerName);

    if contains(class(layer),supportedLayers)
        break
    end

    layerName = findSource(net,layerName);

    if ~isscalar(layerName)
        error("Heads with branches not supported")
    end
end

if isempty(layerName)
    error("Network head not found")
end

layerName = string(layer.Name);

learnableNames = [...
    layerName + "/Weights"
    layerName + "/Bias"];

end

function sourceNames = findSource(net,name)
%FINDSOURCE Find upstream layer
%   sourceNames = findSource(NET,name) returns the names of the layers
%   connected to the specified layer in NET.

connections = net.Connections;
idx = find(connections.Destination == string(name));
sourceNames = connections.Source(idx);

end