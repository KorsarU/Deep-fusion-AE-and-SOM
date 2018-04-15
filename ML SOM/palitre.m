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
    
TwoClassifiersExperiment(results, nnParams)