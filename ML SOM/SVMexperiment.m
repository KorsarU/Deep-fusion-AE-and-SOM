folders = ["Stirling";"ck"; "jaffe"; "mmi"];
labels =  ["angry","disgust","fear","happy","sad","surprise"];
colorl = [1 0 0; 0 1 0; 0 0 1; 1 0 1 ; 0.5 0.5 0.3; 0 0.3 0];
aP = containers.Map;
trainDataSizeCont = containers.Map;
testDataSizeCont = containers.Map;
dataCont = containers.Map;
T = containers.Map;
result = struct;
results = containers.Map;
aeParams = struct;
aeParams.Epochs = 6000;
aeParams.LayersG = [15, 10, 6];
aeParams.LayersL = [200, 150, 100];
aeParams.LayersC = [160, 80, 40];
aeParams.LayersS = [200,150, 75];
aeParams.NumOfLayer = 2;
kernels = ["rbf"; "linear"; "polynomial"];
trainPercantage = 0.85;
stacked = true;
createNew = true;

if createNew
    for i = 1:4
        folder = char(folders(i));
        results(folder) = GetAEfrom(folder, labels, aeParams, aeParams.NumOfLayer>0);
        %here we have four AE, output from AE, and now we can encode other
        %points from different dataset with that AE and calculate its tSNE
    end 
        Smodel2 = results('Stirling') ;
        Cmodel2 = results('ck') ;
        Jmodel2 = results('jaffe');
        Mmodel2 = results('mmi');
else
    if aeParams.NumOfLayer <= 1
    results('Stirling') = Smodel1;
    results('ck') = Cmodel1;
    results('jaffe') = Jmodel1;
    results('mmi') = Mmodel1;
    else
    if aeParams.NumOfLayer == 2
        results('Stirling') = Smodel2;
        results('ck') = Cmodel2;
        results('jaffe') = Jmodel2;
        results('mmi') = Mmodel2;
    else
    if aeParams.NumOfLayer == 3
            results('Stirling') = Smodel3;
            results('ck') = Cmodel3;
            results('jaffe') = Jmodel3;
            results('mmi') = Mmodel3;
    end
    end
    end
end
for i = 1:4
    folder = char(folders(i));
    if aeParams.NumOfLayer == 0
        dataCont(folder) = concatSets(results(folder).SourcePoint.Stacked);
    else
        if stacked
            if aeParams.NumOfLayer == 1
                dataCont(folder) = encode(results(folder).SAE, concatSets(results(folder).SourcePoint.Stacked));
            else
                dataCont(folder) =...
                    results(folder).SAE(...
                    concatSets(results(folder).SourcePoint.Stacked));
            end
        else
            dataCont(folder) = DeepFusionAutoencoder(...
                    concatSets(results(folder).SourcePoint.G),...
                    concatSets(results(folder).SourcePoint.L),...
                    results(folder).GAE,...
                    results(folder).LAE,...
                    results(folder).CAE,...
                    aeParams.NumOfLayer);
        end
    end
    trainDataSizeCont(folder) = 1:(round(size(dataCont(folder),2)*trainPercantage));
    testDataSizeCont(folder) = (round(size(dataCont(folder),2)*trainPercantage)):size(dataCont(folder),2);
    %here we have four AE, output from AE, and now we can encode other
    %points from different dataset with that AE and calculate its tSNE
end 


for k = 1:3
    kernel = char(kernels(k));
    %fprintf('Kernel %s\n): \n\n', kernel);
    Datasets = [];
    Stirling = [];
    Ck = [];
    Jaffe = [];
    MMI = [];
    Average  = [];
    for i = 1:4
        folder = char(folders(i));
        Datasets = [Datasets; folders(i)];
        %fprintf('Train on %s\nTest on:\n', folder);
        data = dataCont(folder);
        trainDataSize = trainDataSizeCont(folder);
        testDataSize = testDataSizeCont(folder);
        Y = ones([1,length(trainDataSize)]); 
        TrainData = data(:,trainDataSize);
        svmmodel = fitcsvm(TrainData', Y', 'KernelFunction', kernel,...
        'KernelScale','auto');

        for j = 1:4
            testFolder = char(folders(j));
            %fprintf('%s: ', testFolder);
            %testData = [mmiData(:,mmiTest), ckdata(:,ckTest), jfData(:,jTest)];
            testData = dataCont(testFolder);
            testtestDataSize = testDataSizeCont(testFolder);
            testData = [...
                testData(:,testtestDataSize),...
                data(:,testDataSize)];
            Y_test = [ones([1,length(testtestDataSize)])+(-2*(i~=j)), ones([1,length(testDataSize)])];
            [~,score] = predict(svmmodel, testData');
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
            if j == 1
                Stirling = [Stirling; acc];
            end
            if j == 2
                Ck = [Ck; acc];
            end
            if j == 3
                Jaffe = [Jaffe; acc];
            end
            if j == 4
                MMI = [MMI; acc];
            end
        end
    end
    Datasets = [Datasets; "Average"];
    Stirling = [Stirling; mean(Stirling)];
    Ck       = [Ck; mean(Ck)];
    Jaffe    = [Jaffe; mean(Jaffe)];
    MMI      = [MMI; mean(MMI)];
    
    for a = 1:5
        mn = mean([Stirling(a), Ck(a), Jaffe(a), MMI(a)]);
        Average = [Average; mn];
    end
        
    T(kernel) = table(Datasets, Stirling, Ck, Jaffe, MMI, Average);
end

for k = 1:3
    fprintf('%s\n\n',kernels(k));
    kernel = char(kernels(k));
    disp(T(kernel));
end

    

    
    
        