%ONE BIG EXPERIMENT

%INITIAL CONDITIONS
aeParams = struct;
aeParams.Epochs = 2000;
aeParams.LayersG = [15, 10, 6];
aeParams.LayersL = [200, 150, 100];
aeParams.LayersC = [330, 100, 50];
aeParams.LayersS = [200,150, 75];
aeParams.LayersH = [1000,500, 250];
aeParams.TypesOfAE = [1,0,1];
aeParams.NumOfLayer = 2;
nnParams = struct;
nnParams.Epochs = 2000;
nnParams.Topol = 'hextop';
nnParams.Size.H = 8;
nnParams.Size.W = 10;
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
%read data from .csvand create initial AE
if true
    models = containers.Map;
    for i = 1:length(DataSetsNames)
        dataset = char(DataSetsNames(i));
        %For generatin stacked on G+L use [1,0,0] TypesofAE aprams, for
        %G+L+H use [1,0,1]
        %For staced [0,1,0]
        models(dataset) = GetAEfrom(dataset, ExpressionLabels, aeParams, needAE);
    end
end


%RUN EXPERIMENT
[Cr, Crn, state] = TwoClassifiersExperiment(models, nnParams);