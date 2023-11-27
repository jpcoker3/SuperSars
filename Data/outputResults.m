function [accuracy] = outputResults(imds,classifier)
predictions = classify(classifier,imds);

YTest = imds.UnderlyingDatastores{1,1}.Labels;

% overall accuracy
accuracy(1,1) = sum(predictions == YTest)/numel(YTest);
% 2S1
accuracy(1,2) = sum(predictions(1:200) == YTest(1:200))/numel(YTest(1:200));
% BRDM_2
accuracy(1,3) = sum(predictions(201:400) == YTest(201:400))/numel(YTest(201:400));
% ZSU_23_4
accuracy(1,4) = sum(predictions(401:600) == YTest(401:600))/numel(YTest(401:600));

end