clear;clc;close all;
%% 读取MNIST数据集
trainTbl = readtable('mnist_train.csv');
testTbl  = readtable('mnist_test.csv');

trainY = trainTbl{:,1};
testY  = testTbl{:,1};
trainRaw = trainTbl{:,2:end};
testRaw  = testTbl{:,2:end};

%% 精细HOG参数，区分1、7顶部短横
cellSz = 3;
% 先获取特征长度，避免维度报错
sampleImg = reshape(trainRaw(1,:),28,28)' / 255;
sampleFeat = extractHOGFeatures(sampleImg, 'CellSize',[cellSz,cellSz]);
featLen = length(sampleFeat);

%% 提取训练集HOG特征
trainFeat = zeros(size(trainRaw,1), featLen);
for i = 1:size(trainRaw,1)
    im = reshape(trainRaw(i,:),28,28)' / 255;
    feat = extractHOGFeatures(im, 'CellSize',[cellSz,cellSz]);
    trainFeat(i,:) = feat;
end

%% 提取测试集HOG特征
testFeat = zeros(size(testRaw,1), featLen);
for i = 1:size(testRaw,1)
    im = reshape(testRaw(i,:),28,28)' / 255;
    feat = extractHOGFeatures(im, 'CellSize',[cellSz,cellSz]);
    testFeat(i,:) = feat;
end

%% 训练多分类SVM
model = fitcecoc(trainFeat, trainY);
save('svm_mnist_model.mat','model');

%% 输出测试准确率
predY = predict(model, testFeat);
acc = mean(predY == testY);
fprintf('SVM手写数字整体识别准确率：%.2f%%\n', acc*100);
