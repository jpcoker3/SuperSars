%% SAR Target Classification on Original MSTAR Dataset 
% SuperSARS - Intro to Radar
% This script serves as the main script for calling the model training
% function. 

clear all; clc;

cnt = 5; % number of times to train each model
subsetAmt = 1000;
trainingAmt = 0.8;
validationAmt = 0.2;

%% Datasets Configuration
cleanDataPath = "F:\Datasets\MSTAR\MSTAR_Dataset"; 
noisy1DataPath = "F:\Datasets\MSTAR\MSTAR_Dataset_Noisy\Gaussian0.1";
noisy2DataPath = "F:\Datasets\MSTAR\MSTAR_Dataset_Noisy\Gaussian0.2";
imDSclean = imageDatastore(cleanDataPath,"LabelSource","foldernames","IncludeSubfolders",true);
imDSnoisy1 = imageDatastore(noisy1DataPath,"LabelSource","foldernames","IncludeSubfolders",true);
imDSnoisy2 = imageDatastore(noisy2DataPath,"LabelSource","foldernames","IncludeSubfolders",true);
% Take 1000 random images from each label. Then, split the dataset into 
% training and validation. Random seed is for reproducability across the
% clean and noisy datasets.
rng(1)
imDSclean = splitEachLabel(imDSclean, subsetAmt, 'randomize');
[cleanTrainSubset, cleanValSubset] = splitEachLabel(imDSclean, trainingAmt, ...
    validationAmt,'randomize');
rng(1)
imDSnoisy1 = splitEachLabel(imDSnoisy1, subsetAmt, 'randomize');
[noisyTrainSubset, noisy1ValSubset] = splitEachLabel(imDSnoisy1, trainingAmt, ...
    validationAmt,'randomize');
rng(1)
imDSnoisy2 = splitEachLabel(imDSnoisy2, subsetAmt, 'randomize');
[~, noisy2ValSubset] = splitEachLabel(imDSnoisy2, trainingAmt, ...
    validationAmt,'randomize');
% Augment the datastores to have only one channel instead of three
% (grayscale), resize depending upon architecture being used.
% Augmenting datastores for Matlab architecture:
sample = read(imDSclean);
imgSize = [size(sample,1) size(sample,2) 1];
cleanTrainDS = transform(cleanTrainSubset,@prepData,'IncludeInfo',true);
cleanValDS = transform(cleanValSubset,@prepData,'IncludeInfo',true);
noisyTrainDS = transform(noisyTrainSubset,@prepData,'IncludeInfo',true); 
noisy1ValDS = transform(noisy1ValSubset,@prepData,'IncludeInfo',true); 
noisy2ValDS = transform(noisy2ValSubset,@prepData,'IncludeInfo',true);

%% 1. Training Clean Model, Validating on Clean Data
% First, we will obtain a baseline by training on the "clean" original 
% data, then validate the model on the original "clean" validation dataset.
% The model we are training is a neural network architecture created
% by Matlab.
whichArch = 1;

fprintf("Experiment 1: Training on clean data (Matlab architecture)...");

for j = 1:cnt
    cleanSARModel1(j) = trainSARclassifier(cleanTrainDS, cleanValDS, whichArch, imgSize);
    reset(cleanTrainDS)
    reset(cleanValDS)
    cleanAccuracy(j,:,1) = outputResults(cleanValDS,cleanSARModel1(j));
    gpuDevice(1)
end

%% 2. Training Noisy (0.1) Model, Validating on Clean Data
% For the second experiment, we are attempting to garner the same accuracy
% or increase the accuracy on the clean dataset by adding noise during 
% training. The hypothesis is that by adding noise to the training data, 
% we are making the classification more robust to additional 
% thermal/speckle noise.
% This noisy data is created by adding Guassian noise with a mean of 0.1 
% and variance of 0.01 to the training dataset.
fprintf("Experiment 2: Training on noisy data (Matlab architecture)...");

for j = 1:cnt
    noisySARModel1(j) = trainSARclassifier(noisyTrainDS, cleanValDS, whichArch, imgSize);
    reset(noisyTrainDS)
    reset(cleanValDS)
    cleanAccuracy(j,:,2) = outputResults(cleanValDS,noisySARModel1(j));
    gpuDevice(1)
end

%% 3. Training Clean Model (AlexNet), Validation on Clean Data
% Well, that didn't work! We see that adding noise to the training 
% dataset reduced the overall performance on the clean validation data 
% with very mixed results. We will try to implement an additional 
% architecture to see if the current architecture is causing the 
% overfitting we are seeing. We will implement AlexNet. Again, we will 
% train the model on clean data as a baseline.
whichArch = 2;
% Augmenting datastores for AlexNet architecture...
cleanTrainDS = transform(cleanTrainSubset,@prepDataArch2,'IncludeInfo',true);
cleanValDS = transform(cleanValSubset,@prepDataArch2,'IncludeInfo',true);
noisyTrainDS = transform(noisyTrainSubset,@prepDataArch2,'IncludeInfo',true); 

fprintf("Experiment 3: Training on clean data (AlexNet)...");

for j = 1:cnt
    cleanSARModel2(j) = trainSARclassifier(cleanTrainDS, cleanValDS, whichArch, imgSize);
    reset(cleanTrainDS)
    reset(cleanValDS)
    cleanAccuracy(j,:,3) = outputResults(cleanValDS,cleanSARModel2(j));
    gpuDevice(1)
end

%% 4. Training Noisy Model (AlexNet), Validation on Clean Data
% Again, using guassian noise with a mean of 0.1 and variance of 0.01...
fprintf("Experiment 4: Training on noisy data (AlexNet)...");

for j = 1:cnt
    noisySARModel2(j) = trainSARclassifier(noisyTrainDS, cleanValDS, whichArch, imgSize);
    reset(noisyTrainDS)
    reset(cleanValDS)
    cleanAccuracy(j,:,4) = outputResults(cleanValDS,noisySARModel2(j));
    gpuDevice(1)
end

%% 5. Validating Clean Model (AlexNet) on Addition of 0.1 and 0.2 Noise
% On the contrary, what happens if we test a clean model on nois(ier) data?
% Will the model still perform well?
for j = 1:cnt
    noisy1Accuracy(j,:,1) = outputResults(noisy1ValDS,cleanSARModel1(j));
    noisy2Accuracy(j,:,1) = outputResults(noisy2ValDS,cleanSARModel1(j));
    noisy1Accuracy(j,:,2) = outputResults(noisy1ValDS,noisySARModel1(j));
    noisy2Accuracy(j,:,2) = outputResults(noisy2ValDS,noisySARModel1(j));
    gpuDevice(1);
end

noisy1ValDS = transform(noisy1ValSubset,@prepDataArch2,'IncludeInfo',true);
noisy2ValDS = transform(noisy2ValSubset,@prepDataArch2,'IncludeInfo',true); 

for j = 1:cnt
    noisy1Accuracy(j,:,3) = outputResults(noisy1ValDS,cleanSARModel2(j));
    noisy2Accuracy(j,:,3) = outputResults(noisy2ValDS,cleanSARModel2(j));
    noisy1Accuracy(j,:,4) = outputResults(noisy1ValDS,noisySARModel2(j));
    noisy2Accuracy(j,:,4) = outputResults(noisy2ValDS,noisySARModel2(j));
end

fig = uifigure('Name','Clean Validation Results From All Models');
for i = 1:4
    avgAcc(i,:) = mean(cleanAccuracy(:,:,i));
end
t = table(avgAcc(:,1),avgAcc(:,2),avgAcc(:,3),avgAcc(:,4),'VariableNames', ...
    ["Total Accuracy","2S1","BRDM_2", "ZSU_23_4"]);
%4x4 matrix
uitable(fig,"Data",t)


fig = uifigure('Name','Noisy (0.1) Validation Results From All Models');
for i = 1:4
    avgAcc(i,:) = mean(noisy1Accuracy(:,:,i));
end
t = table(avgAcc(:,1),avgAcc(:,2),avgAcc(:,3),avgAcc(:,4),'VariableNames', ...
    ["Total Accuracy","2S1","BRDM_2", "ZSU_23_4"]);
uitable(fig,"Data",t)

fig = uifigure('Name','Noisy (0.2) Validation Results From All Models');
for i = 1:4
    avgAcc(i,:) = mean(noisy2Accuracy(:,:,i));
end
t = table(avgAcc(:,1),avgAcc(:,2),avgAcc(:,3),avgAcc(:,4),'VariableNames', ...
    ["Total Accuracy","2S1","BRDM_2", "ZSU_23_4"]);
uitable(fig,"Data",t)

%% Example Classification Output
% Showing an example of SAR target classification using one of the models
% (randomly chosen):

imds = shuffle(noisy2ValDS);
predictions = classify(cleanSARModel2(3),imds);
numImages = 9;
figure
tiledlayout("flow")
for i = 1:numImages
    tmp = read(imds);
    nexttile
    imshow(tmp{1});
    title("Predicted Label: " + string(predictions(i)))
end

% THE END!
% -----------------------------------------------------------------------
%% Helper Functions
% Function to prepare dataset input into model.
function [labelled_img,info] = prepData(img,info)
img = im2gray(img);
img = im2double(img);
labelled_img = {img, info.Label};
end

function [labelled_img,info] = prepDataArch2(img,info)
img = im2gray(img);
img = imresize(img,[227 227]);
img = im2double(img);
labelled_img = {img, info.Label};
end


