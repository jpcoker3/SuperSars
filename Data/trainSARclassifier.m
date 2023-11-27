%% SAR Target Classification Function
% SuperSARS - Intro to Radar

function net = trainSARclassifier(trainDS,valDS,whichArch,imgSize)

    if whichArch == 1
        % Create network using a helper function developed by the Matlab team
        % (See citation below in the Helper Functions section).
        % The neural network is defined within the function. The last layer
        % had to be edited to classify three classes instead of eight.
        % Matlab's neural network:
        layers = createNetwork(imgSize);
        lgraph = layerGraph(layers);
        maxEpochs = 8;
    else
        % Training with AlexNet model
        layers = alexnet('Weights','none');
        lgraph = layerGraph(layers);
        inputLayer = imageInputLayer([227 227 1]);
        lgraph = replaceLayer(lgraph,'data',inputLayer);
        fcLayer = fullyConnectedLayer(3);
        lgraph = replaceLayer(lgraph,'fc8',fcLayer);
        maxEpochs = 16;
    end

    % We used the training parameters that the Matlab team defined in
    % order to have a good baseline to start with. The network architecture
    % has been proven (using these parameters) to train effectively. If we 
    % find that for any reason our model breaks down/doesn't train well, then
    % we will edit these.
    % We changed the minibatchsize to accomodate the computational power of the
    % system that the model is being trained on.
    % It is common practice to validate the neural network at the end of each
    % epoch. Thus, the validation frequency has been changed to (total
    % dataset/minibatchsize).
    miniBatchSize = 16;
    amtOfImages = length(trainDS.UnderlyingDatastores{1}.Files);
    validationFreq = floor(amtOfImages/miniBatchSize);
    options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.001,... %0.001
        'MaxEpochs',maxEpochs, ...
        'Shuffle','every-epoch', ...
        'MiniBatchSize',miniBatchSize,...
        'ValidationData',valDS, ...
        'ValidationFrequency',validationFreq, ...
        'OutputNetwork','best-validation-loss', ...
        'Verbose',false, ...
        'Plots','training-progress');
    
    net = trainNetwork(trainDS,lgraph,options);
    
end

%% Helper Functions
% Function taken from Matlab tutorial on SAR Classification: 
% https://www.mathworks.com/help/radar/ug/sar-target-classification-using-deep-learning.html
function layers = createNetwork(imgSize)
    layers = [
        imageInputLayer([imgSize(1) imgSize(2) 1])      % Input Layer
        convolution2dLayer(3,32,'Padding','same')       % Convolution Layer
        reluLayer                                       % Relu Layer
        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer                         % Batch normalization Layer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)                 % Max Pooling Layer
        
        convolution2dLayer(3,64,'Padding','same')
        reluLayer
        convolution2dLayer(3,64,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        
        convolution2dLayer(3,128,'Padding','same')
        reluLayer
        convolution2dLayer(3,128,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
    
        convolution2dLayer(3,256,'Padding','same')
        reluLayer
        convolution2dLayer(3,256,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
    
        convolution2dLayer(6,512)
        reluLayer
        
        dropoutLayer(0.5)                               % Dropout Layer
        fullyConnectedLayer(512)                        % Fully connected Layer.
        reluLayer
        fullyConnectedLayer(3)
        softmaxLayer                                    % Softmax Layer
        classificationLayer                             % Classification Layer
        ];
end