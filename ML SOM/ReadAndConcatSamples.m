path = 'mmi';
parts = strsplit(pwd, filesep);
currentFolder = parts{end}
if ~(strcmp(currentFolder, path))
    cd(path);
end
hiddenNodes=40;
maxEpochs = 2000;

namesLBPF = ["AngryLBPF.csv", "DisgustLBPF.csv", "FearLBPF.csv", "HappyLBPF.csv", "NeutralLBPF.csv", "SadLBPF.csv", "SurprisedLBPF.csv"]
namesGF = ["AngryGF.csv", "DisgustGF.csv", "FearGF.csv", "HappyGF.csv", "NeutralGF.csv", "SadGF.csv", "SurprisedGF.csv"]

trainSetG = containers.Map;
pg = [];
for i = 1:length(namesGF)
    buf = csvread(namesGF(i));
    b_size = size(buf,1);
    b_dec = int32(fix(b_size/10));
    trainSetG(int2str(i)) = buf(b_size-b_dec+1:b_size,:);
    pg = vertcat(pg, buf(1:(b_size-b_dec),:));
end
pg = myColShuffle(transp(myNorm(pg)));
GFPae = trainAutoencoder(pg,10,...
            'MaxEpochs', maxEpochs*5,...
            'L2WeightRegularization',0.001,...
            'SparsityRegularization',4,...
            'SparsityProportion',0.10);

pgr = predict(GFPae, pg);

'pg-pgr'
mse(pg-pgr)

pgEncoded = encode(GFPae, pg);

trainSetL = containers.Map;
pl = [];
for i = 1:length(namesLBPF)
    buf = csvread(namesLBPF(i));
    b_size = size(buf,1);
    b_dec = int32(fix(b_size/10));
    trainSetL(int2str(i)) = buf(b_size-b_dec+1:b_size,:);
    pl = vertcat(pl, buf(1:(b_size-b_dec),:));
end
pl = myColShuffle(transp(myNorm(pl)));
LBPae = trainAutoencoder(pl,118,...
            'MaxEpochs', 2000,...
            'L2WeightRegularization',0.001,...
            'SparsityRegularization',4,...
            'SparsityProportion',0.10);
plr = predict(LBPae, pl);
'pl-plr'
mse(pl - plr)

plEncoded = encode(LBPae, pl);
pccEncoded = vertcat(plEncoded, pgEncoded);
pccEncoded = myColShuffle(pccEncoded);

pcc = vertcat(plr,pgr);

trainSetC = containers.Map;
for i = 1:length(namesGF)
    id = int2str(i);
    trainSetC(id) = horzcat(trainSetL(id), trainSetG(id));
end
pcc = myColShuffle(pcc);

CCPae = trainAutoencoder(pcc,128,...
            'MaxEpochs', maxEpochs*2,...
            'L2WeightRegularization',0.001,...
            'SparsityRegularization',2,...
            'SparsityProportion',0.5);
       
pccr = predict(CCPae, pcc);
mse(pcc-pccr)
view(CCPae);

CCPae2 = trainAutoencoder(pcc, 60,...
    'MaxEpochs', maxEpochs*2,...
    'L2WeightRegularization', 0.001, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.05, ...
    'DecoderTransferFunction','purelin');



CCPaeForEncoded = trainAutoencoder(pccEncoded, 60, ...
    'L2WeightRegularization', 0.001, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.05, ...
    'DecoderTransferFunction','purelin');

display('pccEncoded');
mse(pccEncoded, predict(CCPaeForEncoded, pccEncoded));


testSet = containers.Map;
mseError = zeros(1, length(trainSetC));
for i = 1:length(trainSetC)
    testOrigin = (transp(myNorm(trainSetC(int2str(i)))));
    testReconstruct = predict(CCPae, testOrigin);
    mseError(i) = mse(testOrigin-testReconstruct);
    testSet(int2str(i)) = testReconstruct;
end
display(mseError);

angry=[];
disgust=[];
fear=[];
happy = [];
neutral = [];
sad = [];
surprised = [];

% for i = 1:7
%      y = network5(testSet(int2str(i)));
%      subplot(2,4,i);
%      classes = vec2ind(y);
%      plot(classes);
%  end

cd('..\')