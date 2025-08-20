%% Load Data for the AlexNet Demo
rootdir = [char(pwd) char('\archive (4)\Plants_2\')];

subdir = [rootdir 'demo'];
demoImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

net = alexnet;

% Anonymous functions for matlab would be like a lambda in python
extractCode = @(s) regexp(s, '\((P\d+)', 'tokens', 'once');
yDemo = categorical(cellfun(@(s) extractCode(s), cellstr(demoImages.Labels)));

demoImages.Labels = yDemo;
inputSize = net.Layers(1).InputSize;

augimdsDemo = augmentedImageDatastore(inputSize(1:2),demoImages);

%% Load in the AlexNet model
load("alexnet.mat")

% Classify the demo images using the AlexNet model
alexNetPredictedLabels = classify(netTransfer, augimdsDemo);

% Display the predicted labels for the demo images
disp(alexNetPredictedLabels);

% Calculate the accuracy of the predictions
trueLabels = demoImages.Labels;
accuracy = sum(alexNetPredictedLabels == trueLabels) / numel(trueLabels);
fprintf('Accuracy of the model on demo images: %.2f%%\n', accuracy * 100);

%% Load Data for the Xception Demo
rootdir = [char(pwd) char('\archive (4)\Plants_2\')];

subdir = [rootdir 'demo'];
demoImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

net = xception;

% Anonymous functions for matlab would be like a lambda in python
extractCode = @(s) regexp(s, '\((P\d+)', 'tokens', 'once');
yDemo = categorical(cellfun(@(s) extractCode(s), cellstr(demoImages.Labels)));

demoImages.Labels = yDemo;
inputSize = net.Layers(1).InputSize;

augimdsDemo = augmentedImageDatastore(inputSize(1:2),demoImages);

%% Load in the Xception model
load("xception.mat")

% Classify the demo images using the AlexNet model
xceptionPredictedLabels = classify(netTransfer, augimdsDemo);

% Display the predicted labels for the demo images
disp(xceptionPredictedLabels);

% Calculate the accuracy of the predictions
trueLabels = demoImages.Labels;
accuracy = sum(xceptionPredictedLabels == trueLabels) / numel(trueLabels);
fprintf('Accuracy of the model on demo images: %.2f%%\n', accuracy * 100);

%% Display Demo images
img1 = imread("archive (4)\Plants_2\demo\Gauva diseased (P3b)\0015_0005.JPG");
img2 = imread("archive (4)\Plants_2\demo\Jamun healthy (P5a)\0005_0005.JPG");
img3 = imread("archive (4)\Plants_2\demo\Pongamia Pinnata diseased (P7b)\0019_0006.JPG");
img4 = imread("archive (4)\Plants_2\demo\Pongamia Pinnata diseased (P7b)\0019_0007.JPG");
img5 = imread("archive (4)\Plants_2\demo\Pongamia Pinnata diseased (P7b)\0019_0008.JPG");
img6 = imread("archive (4)\Plants_2\demo\Pongamia Pinnata diseased (P7b)\0019_0010.JPG");

figure;

subplot(2, 3, 1);
imshow(img1);
title(sprintf("img%d; Alexnet: %s, Xception: %s, Actual: %s", 1, alexNetPredictedLabels(1), xceptionPredictedLabels(1), trueLabels(1)));

subplot(2, 3, 2);
imshow(img2);
title(sprintf("img%d; Alexnet: %s, Xception: %s, Actual: %s", 2, alexNetPredictedLabels(2), xceptionPredictedLabels(2), trueLabels(2)));

subplot(2, 3, 3);
imshow(img3);
title(sprintf("img%d; Alexnet: %s, Xception: %s, Actual: %s", 3, alexNetPredictedLabels(3), xceptionPredictedLabels(3), trueLabels(3)));



subplot(2, 3, 4);
imshow(img4);
title(sprintf("img%d; Alexnet: %s, Xception: %s, Actual: %s", 4, alexNetPredictedLabels(4), xceptionPredictedLabels(4), trueLabels(4)));

subplot(2, 3, 5);
imshow(img5);
title(sprintf("img%d; Alexnet: %s, Xception: %s, Actual: %s", 5, alexNetPredictedLabels(5), xceptionPredictedLabels(5), trueLabels(5)));


subplot(2, 3, 6);
imshow(img6);
title(sprintf("img%d; Alexnet: %s, Xception: %s, Actual: %s", 6, alexNetPredictedLabels(6), xceptionPredictedLabels(6), trueLabels(6)));
