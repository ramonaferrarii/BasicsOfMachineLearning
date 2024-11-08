%% Task 1 : Feedforward multi-layer networks (multi-layer perceptrons)
% 
% clear; clc; close all;
% 
% %Open the Neural Net Pattern Recognition app using
% %nprtool
% 
% % This command loads the predictors glassInputs and the responses glassTargets into the workspace
% load glass_dataset
% 
% % Load the previous quantities in different variables
% [x,t] = glass_dataset;
% 
% %Define training algorithm
% trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
% 
% % Create network
% hiddenLayerSize = 10;
% net = patternnet(hiddenLayerSize, trainFcn);
% 
% % Divide Data
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
% 
% % Train network
% [net,tr] = train(net,x,t);
% 
% % Test network
% y = net(x);
% e = gsubtract(t,y);
% performance = perform(net,t,y)
% 
% tind = vec2ind(t);
% yind = vec2ind(y);
% percentErrors = sum(tind ~= yind)/numel(tind)
% 
% tInd = tr.testInd;
% tstOutputs = net(x(:,tInd));
% tstPerform = perform(net,t(tInd),tstOutputs)
% 
% % View network
% view(net)
% 
% % Plot confusion matrix
% figure, plotconfusion(t,y)

%% Task2 : Autoencoder
clear; clc; close all; 
% Load MNIST data, selecting only two interesting classes
[dataset, target] = loadMNIST(0, [1,8]);

% Number of hidden neurons
nh = 2;

Autoencoder = trainAutoencoder(dataset', nh);
EncodedData = encode(Autoencoder, dataset');
ReconstructedData = predict(Autoencoder, dataset');

% Plot the encoded data using plotcl
figure;
plotcl(EncodedData', target'); 

% Add labels and legend
xlabel('Encoded Feature 1');
ylabel('Encoded Feature 2');
legend('Class 1', 'Class 8');