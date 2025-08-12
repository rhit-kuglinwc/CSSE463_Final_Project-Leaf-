load("leaf_features.mat")

net = xception;

layersTransfer = net.Layers(1:end-3);

% Convert the network to a layerGraph (to preserve connections)
lgraph = layerGraph(net);

% Remove original classification layers
lgraph = removeLayers(lgraph, {'predictions', 'predictions_softmax', 'ClassificationLayer_predictions'});

% Add new classification layers
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc')
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_classification')];

% Connect the last layer from the feature extractor to the new layers
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'new_fc');

% Set training options
options = trainingOptions('adam', ...
    'ValidationData', augimdsValidate, ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 4, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the new network
netTransfer = trainNetwork(augimdsTrain, lgraph, options);

save("xception.mat", "netTransfer")