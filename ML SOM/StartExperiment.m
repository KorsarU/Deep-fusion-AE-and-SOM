function state = StartExperiment(...
    aeEpochs, nnEpochs, topol,...
    numOfLayers, height, width,...
    hiddenLayerG, hiddenLayerL, hiddenLayerC, numOfExp)

state = true;
labels = ["angry","disgust","fear","happy","sad","surprise"];
AESet = containers.Map;
GPAutoencoderSet = containers.Map;
LBPAutoencoderSet = containers.Map;
GpointSet = containers.Map;
LpointSet = containers.Map;
CpointSet = containers.Map;
GTestSet = containers.Map;
LTestSet = containers.Map;
ErrorSet = containers.Map;
TrainSet = containers.Map;
SOMSet = containers.Map;
SOMSetResult = containers.Map;
TrainingLabels = containers.Map;
folders = [ "Stirling";"ck"; "jaffe"; "mmi"];
%aeEpochs = 6000;
%nnEpochs = 4000;
%topol = 'hextop';
%numOfLayers = 3;
%height = 8;
%width = 10;
pathS = "Result/SelfTest "+numOfLayers +"ae "+height+"x"+width+" "+aeEpochs+"ae"+nnEpochs+"ne"+numOfExp+".xlsx"; %path to output
pathC = "Result/CrossTest "+numOfLayers +"ae "+height+"x"+width+" "+aeEpochs+"ae"+nnEpochs+"ne"+numOfExp+".xlsx"; %path to output

%hiddenLayerG = [16, 12, 8];
%hiddenLayerL = [200, 150, 100];
%hiddenLayerC = [90, 55, 30];

if 1==1
for f = 1:length(folders)
    folder = char(folders(f));
    [GpointSet(folder), LpointSet(folder), GTestSet(folder), LTestSet(folder)] = extractPointsFromFolder(folder, labels);
    normalizeSizeOfSets(GpointSet(folder));
    normalizeSizeOfSets(LpointSet(folder));
    [trainingDataG, ~] = concatSets(GpointSet(folder));
    [trainingDataL, tmpLabels] = concatSets(LpointSet(folder));
    trainingDataG = myNorm(trainingDataG, 1); %normalize eqch row by 1
    trainingDataL = myNorm(trainingDataL, 1);
    
    [   AESet(folder),...
        GPAutoencoderSet(folder),...
        LBPAutoencoderSet(folder),...
        ErrorSet(folder),...
        TrainSet(folder)]...
        = LBPGFAutoEncoder(...
        trainingDataG,...
        trainingDataL,...
        aeEpochs,...
        hiddenLayerG,...
        hiddenLayerL,...
        hiddenLayerC,...
        numOfLayers);
    disp(folder);

    
    disp('error:');
    disp(ErrorSet(folder));
    nn = selforgmap([width, height], nnEpochs, 3, topol);
    
    %input =  myNorm(TrainSet(folder),1);
    input =  TrainSet(folder);
    nn = train(nn,input);
    
    SOMSet(folder) = nn;
    TrainingLabels(folder) = tmpLabels;
    SOMSetResult(folder) = plotsomehitsbylabels(nn, input, tmpLabels,height, width, 1, numOfExp+ " " + folder );
end

%MMI+CK+STIRLING
folder = char('MCS');
%[GpointSet(folder), LpointSet(folder), GTestSet(folder), LTestSet(folder)] = 

[mmiPartG,~] = concatSets(GpointSet('mmi'));
[mmiPartL,mmiPartLabels] = concatSets(LpointSet('mmi'));
[ckPartG,~] = concatSets(GpointSet('ck'));
[ckPartL,ckPartLabels] = concatSets(LpointSet('ck'));
[stPartG,~] = concatSets(GpointSet('Stirling'));
[stPartL,stPartLabels] = concatSets(LpointSet('Stirling'));
stPartG = myNorm(stPartG,1); 
ckPartG = myNorm(ckPartG,1);
mmiPartG = myNorm(mmiPartG,1);
stPartL = myNorm(stPartL,1);
ckPartL = myNorm(ckPartL,1);
mmiPartL = myNorm(mmiPartL,1);

GpointSet(folder) =  horzcat(horzcat(stPartG, ckPartG),mmiPartG);
LpointSet(folder) =  horzcat(horzcat(stPartL, ckPartL),mmiPartL);
TrainingLabels(folder) = horzcat(horzcat(stPartLabels, ckPartLabels),mmiPartLabels);
[AESet(folder),GPAutoencoderSet(folder),LBPAutoencoderSet(folder), ErrorSet(folder), TrainSet(folder)] =...
    LBPGFAutoEncoder(...
    GpointSet(folder),LpointSet(folder), ...
    aeEpochs, ...
    hiddenLayerG, hiddenLayerL, hiddenLayerC,...
    numOfLayers);
disp(folder);
disp('error:');
disp(ErrorSet(folder));
nn = selforgmap([width, height],nnEpochs, 3, topol);

input =  TrainSet(folder);
tmpLabels = TrainingLabels(folder);
nn = train(nn,input);
%plotsompos(nn, input); title(folder);
SOMSet(folder) = nn;
SOMSetResult(folder) = plotsomehitsbylabels(nn, input, tmpLabels,height, width, 1, numOfExp+ " " + folder );
%CK+JAFFE
folder = char('CJ');
%[GpointSet(folder), LpointSet(folder), GTestSet(folder), LTestSet(folder)] = 

firstPartG = ckPartG;
firstPartL = ckPartL;
[secondPartG,~] = concatSets(GpointSet('jaffe'));
[secondPartL,jaffePartLabels] = concatSets(LpointSet('jaffe'));
jaffePartG = secondPartG;
jaffePartL = secondPartL;
secondPartG = myNorm(secondPartG,1);
%firstPartG  = myNorm(firstPartG ,1);

secondPartL = myNorm(secondPartL,1);
%firstPartL = myNorm(firstPartL ,1);
GpointSet(folder) =  horzcat(firstPartG, secondPartG );
LpointSet(folder) =  horzcat(firstPartL, secondPartL );
TrainingLabels(folder)=horzcat(ckPartLabels, jaffePartLabels);
[AESet(folder),GPAutoencoderSet(folder),LBPAutoencoderSet(folder), ErrorSet(folder), TrainSet(folder)] =...
    LBPGFAutoEncoder(...
    GpointSet(folder),LpointSet(folder), ...
    aeEpochs, ...
    hiddenLayerG, hiddenLayerL, hiddenLayerC,...
    numOfLayers);
disp(folder);
disp('error:');
disp(ErrorSet(folder));
nn = selforgmap([width, height],nnEpochs, 3, topol);

input =  TrainSet(folder);
tmpLabels = TrainingLabels(folder);
nn = train(nn,input);
%plotsompos(nn, input); title(folder);
SOMSet(folder) = nn;
SOMSetResult(folder) = plotsomehitsbylabels(nn, input, tmpLabels,height, width, 1, folder+ " " +numOfExp);

%MMI+JAFFE
folder = char('MJ');
%[GpointSet(folder), LpointSet(folder), GTestSet(folder), LTestSet(folder)] = 
firstPartG = mmiPartG;
firstPartL = mmiPartL;
secondPartG = jaffePartG;
secondPartL = jaffePartL;
tmpLabels = horzcat(mmiPartLabels, jaffePartLabels);
GpointSet(folder) =  horzcat(firstPartG, secondPartG );
LpointSet(folder) =  horzcat(firstPartL, secondPartL );
TrainingLabels(folder) = tmpLabels;
[AESet(folder),GPAutoencoderSet(folder),LBPAutoencoderSet(folder), ErrorSet(folder), TrainSet(folder)] =...
    LBPGFAutoEncoder(...
    GpointSet(folder),LpointSet(folder), aeEpochs,...
    hiddenLayerG, hiddenLayerL, hiddenLayerC,...
    numOfLayers);
disp(folder);
disp('error:');
disp(ErrorSet(folder));
nn = selforgmap([width, height],nnEpochs, 3, topol);

%input =  myNorm(TrainSet(folder),1);
input =  TrainSet(folder);
nn = train(nn,input);
%plotsompos(nn, input); title(folder);
SOMSet(folder) = nn;
SOMSetResult(folder) = plotsomehitsbylabels(nn, input, tmpLabels,height, width, 1, numOfExp+ " " + folder );
end

%Prepare TestDataSet
K = GTestSet.keys;
TestDataSet = containers.Map;

for i = 1:4
    K_id = char(K(i));
    gts = GTestSet(K_id);
    lts = LTestSet(K_id);
    k = gts.keys;
    testSet = containers.Map;
    ...disp(K_id);
    gpae = GPAutoencoderSet(K_id);
    lbpae = LBPAutoencoderSet(K_id);
    cae = AESet(K_id);
    for j = 1:length(k)
        exT = char(k(j));
    ...    disp(exT);
        graphP = (transp(myNorm(gts(exT),2)));%encode(GPAutoencoderSet(K_id), transp(myNorm(gts(exT),2)));
        lbpP = (transp(myNorm(lts(exT),2)));%encode(LBPAutoencoderSet(K_id), transp(myNorm(lts(exT),2)));
        testSet(exT) = DeepFusionAutoencoder(graphP,lbpP,gpae,lbpae,cae,numOfLayers);%(encode(AESet(K_id), cc));
    ...    disp(size(testSet(exT(1,:))));
    end
    TestDataSet(K_id) = testSet;
end

%MCS
testSet = containers.Map;
for i = 1:length(k)
    exT = char(k(i));
    m_i = TestDataSet('mmi');
    m_i = m_i(exT);
    c_i = TestDataSet('ck');
    c_i = c_i(exT);
    s_i = TestDataSet('Stirling');
    s_i = s_i(exT);
    testSet(exT) = horzcat(horzcat(m_i, c_i),s_i);
end
TestDataSet('MCS') = testSet;

%MJ
testSet = containers.Map;
for i = 1:length(k)
    exT = char(k(i));
    m_i = TestDataSet('mmi');
    m_i = m_i(exT);
    j_i = TestDataSet('jaffe');
    j_i = j_i(exT);
    testSet(exT) = horzcat(m_i,j_i);
end
TestDataSet('MJ') = testSet;

%CJ
testSet = containers.Map;
for i = 1:length(k)
    exT = char(k(i));
    c_i = TestDataSet('ck');
    c_i = c_i(exT);
    j_i = TestDataSet('jaffe');
    j_i = j_i(exT);
    testSet(exT) = horzcat(c_i,j_i);
end

TestDataSet('CJ') = testSet;

%SampleHitsSet = containers.Map;
NNsetKeys = SOMSet.keys;
...disp('neurons distribution');
%Maybe switch off that part
...for i = 1:length(NNsetKeys)
...    f = char(NNsetKeys(i));
...    if ~(strcmp(f, 'MCS') || strcmp(f, 'CJ') || strcmp(f, 'MJ'))
...    nn = SOMSet(f);
...    disp(f);
...   % h = plotsomhits(nn, TrainSet(f)); title(f);
...%     waitfor(h);
...    gp = GpointSet(f);
...    lp = LpointSet(f);
...   % buf = containers.Map;
    
...    for j = 1:length(labels)
...        exT = char(labels(j));
...        ccp = DeepFusionAutoencoder(gp(exT),...
...            lp(exT),...
...            GPAutoencoderSet(f),...
...            LBPAutoencoderSet(f),...
...            AESet(f), numOfLayers);
        %h = plotsomhits(nn,ccp); title(exT);
        %waitfor(h);
        %buf(exT) = h;
...        v1 = vec2ind(nn(ccp));
...        vu = unique(v1);
...        str = [length(v1), exT]; 
...        disp(str);
...        disp(vu);
...        str=[];
...        for u = 1:length(vu)
...            str = [str, sum(v1==vu(u))];
...        end
...        disp(str);
...    end
    %SampleHitsSet(f) = buf;
...    end
...end


% (1) Train on MMI and test on MMI
% (2) Train on MMI and test on CK
% (3) Train on CK and test on CK
% (4) Train on CK and test on MMI
% (5) Train on Stirling and test on Stirling
% (6) Train on Stirling and test on CK
% (7) Train on MMI+CK+Stirling and test on MMI+CK+Stirling
% (8) Train on MMI+CK+Stirling and test on JAFFE
% (8) Train on JAFFE and test on JAFFE
% (8) Train on JAFFE and test on MMI


SbSSet = containers.Map;
disp('Test selfBySelf');
f_k = GpointSet.keys;
for i = 1:length(f_k)
    f = char(f_k(i));
    somSetRes = SOMSetResult(f);
    testSet = TestDataSet(f);
    nn = SOMSet(f);
    disp(f);
    SbSSet(f) =  CalculateResult(nn, testSet, k, height,width, somSetRes);
end
    
disp('cross test')

CrossTestSet = containers.Map; %keyParameter = TrainDatasetName + TestDatasetName

disp('mmi with ...')
f = 'mmi';
testSet = TestDataSet('ck');
nn = SOMSet(f);
somSetRes = SOMSetResult(f);
disp('mmi test with ck')
CrossTestSet('mmi+ck') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('mmi test with ck+jaffe')
testSet = TestDataSet('CJ');
CrossTestSet('mmi+cj') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('mmi test with Jaffe')
testSet = TestDataSet('jaffe');
CrossTestSet('mmi+jaffe') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('mmi test with stirling')
testSet = TestDataSet('Stirling');
CrossTestSet('mmi+Stirling') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('ck with...')
f = 'ck';
nn = SOMSet(f);
somSetRes = SOMSetResult(f);

disp('ck test with jaffe');
testSet = TestDataSet('jaffe');
CrossTestSet('ck+jaffe') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('ck test with mmi');
testSet = TestDataSet('mmi');
CrossTestSet('ck+mmi') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('ck test with MJ');
testSet = TestDataSet('MJ');
CrossTestSet('ck+MJ') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('ck test with Stirling');
testSet = TestDataSet('Stirling');
CrossTestSet('ck+Stirling') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('Stirlin with ...')
f = 'Stirling';
somSetRes = SOMSetResult(f);
nn = SOMSet(f);

disp('Stirling test with ck');
testSet = TestDataSet('ck');
CrossTestSet('Stirling+ck') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('Stirling test with jaffe');
testSet = TestDataSet('jaffe');
CrossTestSet('Stirling+jaffe') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('Stirling test with mmi');
testSet = TestDataSet('mmi');
CrossTestSet('Stirling+mmi') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('Stirling test with MJ');
testSet = TestDataSet('MJ');
CrossTestSet('Stirling+MJ') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('Jaffe with ...')
f = 'jaffe';
somSetRes = SOMSetResult(f);
nn = SOMSet(f);

disp('jaffe test with ck');
testSet = TestDataSet('ck');
CrossTestSet('jaffe+ck') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('jaffe test with MCS');
testSet = TestDataSet('MCS');
CrossTestSet('jaffe+MCS') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('jaffe test with mmi');
testSet = TestDataSet('mmi');
CrossTestSet('jaffe+mmi') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('jaffe test with Stirling');
testSet = TestDataSet('Stirling');
CrossTestSet('jaffe+Stirling') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('MCS with ...');
f = 'MCS';
nn = SOMSet(f);
somSetRes = SOMSetResult(f);

disp('MCS test with ck');
testSet = TestDataSet('ck');
CrossTestSet('MCS+ck') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('MCS test with mmi');
testSet = TestDataSet('mmi');
CrossTestSet('MCS+mmi') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('MCS test with jaffe');
testSet = TestDataSet('jaffe');
CrossTestSet('MCS+jaffe') = CalculateResult(nn,testSet, k, height, width, somSetRes);

disp('MCS test with Stirling');
testSet = TestDataSet('Stirling');
CrossTestSet('MCS+Stirling') = CalculateResult(nn,testSet, k, height, width, somSetRes);


disp('CJ with ...');
f = 'CJ';
nn = SOMSet(f);
somSetRes = SOMSetResult(f);

disp('CJ test with MCS');
testSet = TestDataSet('MCS');
CrossTestSet('CJ+MCS') = CalculateResult(nn,testSet, k, height, width, somSetRes);

testSet = TestDataSet('MJ');
disp('CJ test with MJ');
CrossTestSet('CJ+MJ') = CalculateResult(nn,testSet, k, height, width, somSetRes);

testSet = TestDataSet('mmi');
disp('CJ test with mmi');
CrossTestSet('CJ+mmi') = CalculateResult(nn,testSet, k, height, width, somSetRes);

testSet = TestDataSet('Stirling');
disp('CJ test with Stirling');
CrossTestSet('CJ+Stirling') = CalculateResult(nn,testSet, k, height, width, somSetRes);


disp('MJ with ...');
f = 'MJ';
nn = SOMSet(f);
somSetRes = SOMSetResult(f);

testSet = TestDataSet('ck');
disp('MJ test with ck');
CrossTestSet('MJ+ck') = CalculateResult(nn,testSet, k, height, width, somSetRes);

testSet = TestDataSet('MCS');
disp('MJ test with MCS');
CrossTestSet('MJ+MCS') = CalculateResult(nn,testSet, k, height, width, somSetRes);

testSet = TestDataSet('mmi');
disp('MJ test with mmi');
CrossTestSet('MJ+mmi') = CalculateResult(nn,testSet, k, height, width, somSetRes);


testSet = TestDataSet('Stirling');
disp('MJ test with Stirling');
CrossTestSet('MJ+Stirling') = CalculateResult(nn,testSet, k, height, width, somSetRes);


disp("SELF BY SELF RESULT")


results = [SbSSet.keys; SbSSet.values];
for res = results
    disp(res(1));
    c = res(2);
    %celldisp(c);
    xlswrite(pathS,c{:}, res{1});
    ac = CalcAcc(c{:});
    xlswrite(pathS,"Accuracy",res{1},"E1");
    xlswrite(pathS,ac,res{1},"E2");
end


disp("CROSS TEST RESULT")

results = [CrossTestSet.keys; CrossTestSet.values];
for res = results
    disp(res(1));
    c = res(2);
    %celldisp(c);
    xlswrite(pathC,c{:},res{1});
    ac = CalcAcc(c{:});
    xlswrite(pathC,"Accuracy",res{1},"E1");
    xlswrite(pathC,ac,res{1},"E2");
end


CloseAllFigures();
end