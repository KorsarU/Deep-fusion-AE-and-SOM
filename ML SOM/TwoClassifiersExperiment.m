%selected classififcation
function state = TwoClassifiersExperiment(models,nnParams)
%initial conditions
%models = results;

SOMSet = containers.Map;
SOMSetResult = containers.Map;
TrainingLabels = containers.Map;
CResults = containers.Map;

SOMSetNoised = containers.Map;
SOMSetResultNoised = containers.Map;
TrainingLabelsNoised = containers.Map;
CResultsNoised = containers.Map;

SVMmodels = containers.Map;

aP = containers.Map;
trainDataSizeCont = containers.Map;
testDataSizeCont = containers.Map;
dataCont = containers.Map;
T = containers.Map;
%results = containers.Map;

%aeParams = struct;
%aeParams.Epochs = 6000;
%aeParams.LayersG = [15, 10, 6];
%aeParams.LayersL = [200, 150, 100];
%aeParams.LayersC = [160, 80, 40];
%aeParams.LayersS = [200,150, 75];
%aeParams.NumOfLayer = 2;



%nnParams = struct;
%nnParams.Epochs = 4000;
%nnParams.Topol = 'hextop';
%nnParams.Size.H = 8;
%nnParams.Size.W = 10;


kernel = "rbf";
trainPercantage = 0.85;
numOfExp = 1;
stacked = false;
createNew = false;
needDraw = false;

modelsLenght = length(models);
modelsLenghtRange = 1:modelsLenght;
modelsKeys = models.keys;
    %generate noised data after AE applying
    for i = modelsLenghtRange
        key = char(modelsKeys(i));
        orData = models(key).ResultPoint;
        model = models(key);
        model.SourcePoint.NoisedFusedPoints = AddNoise(orData, mean2(orData));
        models(key) = model;
    end
    
    %train first SOM
    for i = modelsLenghtRange
        key = char(modelsKeys(i));
        model = models(key);
        nn = selforgmap([nnParams.Size.W, nnParams.Size.H], nnParams.Epochs, 3, nnParams.Topol);
        input = model.ResultPoint;
        nn = train(nn,input);
        
        [~, tmpLabels] = concatSets(model.SourcePoint.G);
        
        SOMSet(key) = nn;
        TrainingLabels(key) = tmpLabels;
        SOMSetResult(key) = plotsomehitsbylabels(...
            nn, ...
            input, ...
            tmpLabels,...
            nnParams.Size.H, ...
            nnParams.Size.W, ...
            needDraw, ...
            numOfExp+ " " + key );
    end
    
    %train second SOM
    for i = modelsLenghtRange
        key = char(modelsKeys(i));
        model = models(key);
        nn = selforgmap([nnParams.Size.W, nnParams.Size.H], nnParams.Epochs, 3, nnParams.Topol);
        input = model.SourcePoint.NoisedFusedPoints;
        nn = train(nn,input);
        
        [~, tmpLabels] = concatSets(model.SourcePoint.G);
        
        SOMSetNoised(key) = nn;
        TrainingLabelsNoised(key) = tmpLabels;
        SOMSetResultNoised(key) = plotsomehitsbylabels(...
            nn, ...
            input, ...
            tmpLabels,...
            nnParams.Size.H, ...
            nnParams.Size.W, ...
            needDraw, ...
            numOfExp+ " " + key );
    end
    
    %train SVM
    
    for i = modelsLenghtRange
        key = char(modelsKeys(i));
        model = models(key);
        aeParams = model.aeParams;
        if aeParams.NumOfLayer == 0
            dataCont(key) = concatSets(model.SourcePoint.Stacked);
        else
            if stacked
                if aeParams.NumOfLayer == 1
                    dataCont(key) = encode(model.SAE, concatSets(model.SourcePoint.Stacked));
                else
                    dataCont(key) =...
                        model.SAE(...
                        concatSets(model.SourcePoint.Stacked));
                end
            else
                dataCont(key) = DeepFusionAutoencoder(...
                        concatSets(model.SourcePoint.G),...
                        concatSets(model.SourcePoint.L),...
                        model.GAE,...
                        model.LAE,...
                        model.CAE,...
                        aeParams.NumOfLayer);
            end
        end
        trainDataSizeCont(key) = 1:(round(size(dataCont(key),2)*trainPercantage));
        testDataSizeCont(key) = (round(size(dataCont(key),2)*trainPercantage)):size(dataCont(key),2);
    end 
    
    for i = modelsLenghtRange
        key = char(modelsKeys(i));
        
        %fprintf('Train on %s\nTest on:\n', folder);
        data = dataCont(key);
        trainDataSize = trainDataSizeCont(key);
        testDataSize = testDataSizeCont(key);
        Y = ones([1,length(trainDataSize)]); 
        TrainData = data(:,trainDataSize);
        svmmodel = fitcsvm(TrainData', Y', 'KernelFunction', char(kernel),...
        'KernelScale','auto');

        data = data(:,testDataSize);
        Y_test = ones([1,length(testDataSize)]);
        [~,score] = predict(svmmodel, data');
        labAndScore = [Y_test', score];
        check = zeros([length(Y_test), 1]);
        for f = 1:length(Y_test)
            if labAndScore(f,1) < 0 && labAndScore(f,2) < 0
                check(f) = 1; %True negative
            else
                if labAndScore(f,1) > 0 && labAndScore(f,2) > 0
                    check(f) = 2; % True positive
                else
                    if labAndScore(f,1) > 0 && labAndScore(f,2) < 0
                        check(f) = -1; %False negative
                    else
                        if labAndScore(f,1) < 0 && labAndScore(f,2) > 0
                            check(f) = -2; %False positive
                        end
                    end
                end
            end
        end
        TP = length(check(check==2));
        TN = length(check(check==1));
        FP = length(check(check==-2));
        FN = length(check(check==-1));
        rej = length(score(score<0));
        all = length(Y_test);
        acc = round(((TP+TN)/all)*100,2);
        %disp(1-abs(((length(Y_test)-rej)/length(Y_test)) - (length(mmiTest)/length(Y_test)) ));
        %if i == j
        %    acc = round((1-rej/all)*100,2);
        %else
        %    acc = round((1-abs( ((all-rej-length(testDataSize))/all))*100),2);
        %end
        %fprintf('%.2f%%\n', acc);
        SVMmodels(key) = svmmodel;
    end
 
    
    %Evaluate performance with and without SVM selection
    
   
    
    %Evaluation with SVMS
    
    for i = modelsLenghtRange
        key = char(modelsKeys(i));
        model = models(key);
        aeParams = model.aeParams;
        svmmodel = SVMmodels(key);
        
        for j = modelsLenghtRange
            testKey = char(modelsKeys(j));
            if i == j
                testModel = model;
            else
                testModel = models(testKey);
            end
            TG = testModel.SourcePoint.TG;
            TL = testModel.SourcePoint.TL;
            labels = TG.keys;
            testSet = containers.Map;
            for li = 1:length(labels)
                label = char(labels(li));
                graphP = (transp(myNorm(TG(label),2)));
                lbpP = (transp(myNorm(TL(label),2)));
                testSet(label) = DeepFusionAutoencoder(...
                    graphP,...
                    lbpP,...
                    model.GAE,...
                    model.LAE,...
                    model.CAE,...
                    aeParams.NumOfLayer);
            end
            
            labels = testSet.keys;
            C = concatSets(testSet);
            
            [~,score] = predict(svmmodel, C');
            %fprintf('%s\n',key);
            %if length(score(score<0))>(length(score)/2)
                nn = SOMSetNoised(key);
                somSetRes = SOMSetResultNoised(key);
                CResultsNoised([key,' ', testKey]) =  CalculateResult(...
                    nn, testSet, labels, ...
                    nnParams.Size.H,nnParams.Size.W, somSetRes);
            %else
                nn = SOMSet(key);
                somSetRes = SOMSetResult(key);
                CResults([key,' ', testKey]) =  CalculateResult(...
                    nn, testSet, labels, ...
                    nnParams.Size.H,nnParams.Size.W, somSetRes);
            %end
            
        end
        
    end
    
    %Evaluate without SVMS
    
    for i = modelsLenghtRange
        key = char(modelsKeys(i));
        model = models(key);
        aeParams = model.aeParams;
        for j = modelsLenghtRange
            testKey = char(modelsKeys(j));
            if i == j
                testModel = model;
            else
                
                testModel = models(testKey);
            end
            TG = testModel.SourcePoint.TG;
            TL = testModel.SourcePoint.TL;
            labels = TG.keys;
            testSet = containers.Map;
            for li = 1:length(labels)
                label = char(labels(li));
                graphP = (transp(myNorm(TG(label),2)));
                lbpP = (transp(myNorm(TL(label),2)));
                testSet(label) = DeepFusionAutoencoder(...
                    graphP,...
                    lbpP,...
                    model.GAE,...
                    model.LAE,...
                    model.CAE,...
                    aeParams.NumOfLayer);
            end
            
            labels = testSet.keys;
            
            %fprintf('%s\n',key);
            if ~(CResults.isKey([key, ' ', testKey]))
                nn = SOMSet(key);
                somSetRes = SOMSetResult(key);
                CResults([key, ' ', testKey]) =  CalculateResult(...
                    nn, testSet, labels, ...
                    nnParams.Size.H,nnParams.Size.W, somSetRes);
            end
            
        end
        
    end
    
    results = [CResults.keys; CResults.values];
    disp('Without 2 NN');
    for res = results
  
        c = res(2);
        %celldisp(c);
        %xlswrite(pathS,c{:}, res{1});
        ac = CalcAcc(c{:});
        %xlswrite(pathS,"Accuracy",res{1},"E1");
        %xlswrite(pathS,ac,res{1},"E2");
        %celldisp(res(1));
        s = string(res(1));
        fprintf('%s: %f%%\n', [s, ac]);
    end
    
    disp('Noised NN');
    results = [CResultsNoised.keys; CResultsNoised.values];
    for res = results
  
        c = res(2);
        %celldisp(c);
        %xlswrite(pathS,c{:}, res{1});
        ac = CalcAcc(c{:});
        %xlswrite(pathS,"Accuracy",res{1},"E1");
        %xlswrite(pathS,ac,res{1},"E2");
        %celldisp(res(1));
        s = string(res(1));
        fprintf('%s: %f%%\n', [s, ac]);
    end
    state = true;
end