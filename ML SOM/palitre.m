aeParams = struct;
aeParams.Epochs = 6000;
aeParams.LayersG = [15, 10, 6];
aeParams.LayersL = [200, 150, 100];
aeParams.LayersC = [160, 80, 40];
aeParams.LayersS = [200,150, 75];
aeParams.NumOfLayer = 2;

nnParams = struct;
nnParams.Epochs = 2000;
nnParams.Topol = 'hextop';
nnParams.Size.H = 8;
nnParams.Size.W = 10;
folders = ["Stirling";"ck"; "jaffe"; "mmi"];
labels =  ["angry","disgust","fear","happy","sad","surprise"];
test = false;

if test
    for i = 1:4
    folder = char(folders(i));
    results(folder) = GetAEfrom(folder, labels, aeParams, aeParams.NumOfLayer>0);
    %here we have four AE, output from AE, and now we can encode other
    %points from different dataset with that AE and calculate its tSNE
    end 
end
    
%TwoClassifiersExperiment(results, nnParams)

%plot T-sne for noised and original data for each DS
if false
    path =  "Result\NoisedVsUNoised\";


    for i = 1:length(folders)
        strs = strings(0);
        key = char(folders(i));
        model = models(key);
        fig = figure;
        pointsWON = model.ResultPoint;
        pointsN = model.SourcePoint.NoisedFusedPoints;
        fprintf("noised   %.4f mean, %.4f variance\noriginal %.4f mean, %.4f variance\n",...
            [mean2(pointsN), std2(pointsN), mean2(pointsWON), std2(pointsWON)]);
        sz = size(pointsN);
        sz = sz(2);
        for j = 1:sz
            strs = [strs, "noised"];
        end
        for j = 1:sz
            strs = [strs, "original"];
        end
        strs = strs';
        points = [pointsN, pointsWON];
        [y, loss] = tsne(points');
        fig = gscatter(y(:,1), y(:,2), strs);
        title("Noised vs unnoised: " + model.Name);

    end
end

%Comparison t-sne different ds
if false
    algorithms = [2];%0-with factor, 1 - ussual, 2 - shift, 3 - random
    variances = [1, 1.5, 0.5];
    means = [-1];
    pointsO1 = model1.ResultPoint;
    pointsO2 = model2.ResultPoint;
    pointsO3 = model3.ResultPoint;
    pointsO4 = model4.ResultPoint;

    
    for i = 1:length(means)
        mean_ = means(i);
        for j = 1:length(variances)
            variance_ = variances(j);
            for a = 1:length(algorithms)
                algorithm_ = algorithms(a);
                strs = strings(0);
                
                fig = figure;
                pointsN1 = AddNoise(pointsO1, [mean_, variance_, algorithm_]);%model1.SourcePoint.NoisedFusedPoints;
                pointsN2 = AddNoise(pointsO2, [mean_, variance_, algorithm_]);%model2.SourcePoint.NoisedFusedPoints;
                pointsN3 = AddNoise(pointsO3, [mean_, variance_, algorithm_]);%model2.SourcePoint.NoisedFusedPoints;
                pointsN4 = AddNoise(pointsO4, [mean_, variance_, algorithm_]);%model2.SourcePoint.NoisedFusedPoints;
                
                fprintf("Stirling:\nnoised   %.4f mean, %.4f variance\noriginal %.4f mean, %.4f variance\n",...
                    [mean2(pointsN1), std2(pointsN1), mean2(pointsO1), std2(pointsO1)]);
                fprintf("ck:\nnoised   %.4f mean, %.4f variance\noriginal %.4f mean, %.4f variance\n",...
                    [mean2(pointsN2), std2(pointsN2), mean2(pointsO2), std2(pointsO2)]);
                fprintf("jaffe:\nnoised   %.4f mean, %.4f variance\noriginal %.4f mean, %.4f variance\n",...
                    [mean2(pointsN3), std2(pointsN3), mean2(pointsO3), std2(pointsO3)]);
                fprintf("mmi:\nnoised   %.4f mean, %.4f variance\noriginal %.4f mean, %.4f variance\n\n",...
                    [mean2(pointsN4), std2(pointsN4), mean2(pointsO4), std2(pointsO4)]);
    
                sz = size(pointsN1);
                sz = sz(2);
                for j = 1:sz
                    strs = [strs, "noised Stirling"];
                end
                for j = 1:sz
                    strs = [strs, "original Stirling"];
                end

                sz = size(pointsN2);
                sz = sz(2);
                for j = 1:sz
                    strs = [strs, "noised ck"];
                end
                for j = 1:sz
                    strs = [strs, "original ck"];
                end
                
                sz = size(pointsN3);
                sz = sz(2);
                for j = 1:sz
                    strs = [strs, "noised jaffe"];
                end
                for j = 1:sz
                    strs = [strs, "original jaffe"];
                end
                
                sz = size(pointsN4);
                sz = sz(2);
                for j = 1:sz
                    strs = [strs, "noised mmi"];
                end
                for j = 1:sz
                    strs = [strs, "original mmi"];
                end

                strs = strs';
                points = [pointsN1, pointsO1, pointsN2, pointsO2, ...
                    pointsN3, pointsO3, pointsN4, pointsO4];
                [y, loss] = tsne(points');
                fig = gscatter(y(:,1), y(:,2), strs);
                params = "E= "+string(mean_)+"; V= "+...
                    string(variance_)+"; Alg: "+string(algorithm_);         
                title("Nsd vs unNsd: "+params);
            end
        end
    end
end


if false
    pointsO2 = model2.SourcePoint.Stacked;
    pointsO3 = model3.SourcePoint.Stacked;
    fig = figure;
    if true
        pointsO2 = containers.Map;
        pointsO3 = containers.Map;
        pointsG = model2.SourcePoint.G;
        pointsL = model2.SourcePoint.L;
        pointsH = model2.SourcePoint.H;
        for i = 1:6
            label = char(labels(i));
            pointsO2([label, ' ck']) = [pointsG(label); pointsL(label); pointsH(label)];
        end

        pointsG = model3.SourcePoint.G;
        pointsL = model3.SourcePoint.L;
        pointsH = model3.SourcePoint.H;
        for i = 1:6
            label = char(labels(i));
            pointsO3([label, ' jaffe']) = [pointsG(label); pointsL(label); pointsH(label)];
        end
        

        strs = strings(0);
        [pointsO2, labelsO2] = concatSets(pointsO2);
        [pointsO3, labelsO3] = concatSets(pointsO3);

        sz = size(pointsO2);
        sz = sz(2);
        for j = 1:sz
            strs = [strs, labelsO2(j)];
        end

        sz = size(pointsO3);
        sz = sz(2);
        for j = 1:sz
            strs = [strs, labelsO3(j)];
        end
        strs = strs';
        points = [pointsO2, pointsO3];
        [y, loss] = tsne(points');
        fig = gscatter(y(:,1), y(:,2), strs);
    title("Internal class distribution");
    end
end

folds_num = 5;
if true
    %model1 = models('Stirling');
    %model2 = models('ck');
    %model3 = models('jaffe');
    %model4 = models('mmi');

    pointsOs = model1.ResultPoint;
    %pointsNs = model1.SourcePoint.NoisedFusedPoints;
    [~ , targetsS] = concatSets(model1.SourcePoint.H);
    dataForSVMS = prepareDataForSVMFFromMatrixAndLabels(pointsOs, targetsS,folds_num);
        
    pointsOc = model2.ResultPoint;
    %pointsNc = model2.SourcePoint.NoisedFusedPoints;
    [~ , targetsC] = concatSets(model2.SourcePoint.H);
    dataForSVMC = prepareDataForSVMFFromMatrixAndLabels(pointsOc, targetsC,folds_num);
    
    pointsOj = model3.ResultPoint;
    %pointsNj = model3.SourcePoint.NoisedFusedPoints;
    [~ , targetsJ] = concatSets(model3.SourcePoint.H);
    dataForSVMJ = prepareDataForSVMFFromMatrixAndLabels(pointsOj, targetsJ,folds_num);
    
    pointsOm = model4.ResultPoint;
    %pointsNm = model4.SourcePoint.NoisedFusedPoints;
    [~ , targetsM] = concatSets(model4.SourcePoint.H);
    dataForSVMM = prepareDataForSVMFFromMatrixAndLabels(pointsOm, targetsM,folds_num);
    
    result = 0;
    %[result, SVM] = testSVMwithValidation(dataForSVMJ);
    
    
end
