%ONE BIG EXPERIMENT

%INITIAL CONDITIONS
aeParams = struct;
aeParams.Epochs = 3000;
aeParams.LayersG = [15, 10, 6];
aeParams.LayersL = [200, 150, 100];
aeParams.LayersC = [160, 80, 40];
aeParams.LayersS = [200,150, 75];
aeParams.LayersH = [1000,500, 250];
aeParams.TypesOfAE = [1,1,1];
aeParams.NumOfLayer = 2;

needAE = true;

%GENERATE LIST OF SETTING
featuresNames = ["LBP", "GP", "HOG"];
aeTypes = ["Sparce", "Variational"];
aeNumOfLayers = [2,3];
classifiersTypes = ["SOM", "SVM", "FFN"];
TrainingDataSize = 0.8;
DataSetsNames = ["Stirling";"ck"; "jaffe"; "mmi"];
ExpressionLabels =  ["angry","disgust","fear","happy","sad","surprise"];
PipeLine = ["withoutAE", "withAE"];
NoiseTypes = ["Gauss", "Masking", "Salt-n-Paper"];


%Create all needed elements
if true
    models = containers.Map;
end
for i = 1:length(DataSetsNames)
    dataset = char(DataSetsNames(i));
    models(dataset) = GetAEfrom(dataset, ExpressionLabels, aeParams, needAE);
    
end

%RUN EXPERIMENT