keys = [11,12,13,14,15,16,17,18,19];
    aeEpoch = 2000;
    nnEpoch = 2000;
    topol = 'hextop';
for i = 1:3
    %aeEpoch = 6000;
    %nnEpoch = 4000;
    %topol = 'hextop';
    numOfLayer = i;
    height = 12;
    width = 15;
    hiddenLayerG = [16, 12, 8];
    hiddenLayerL = [200, 150, 100];
    hiddenLayerC = [90, 55, 30];

    StartExperiment(aeEpoch, nnEpoch, topol,...
        numOfLayer, height, width,...
        hiddenLayerG, hiddenLayerL, hiddenLayerC, keys(i));
end

for i = 1:3
    %aeEpoch = 1;
    %nnEpoch = 1;
    %topol = 'hextop';
    numOfLayer = i;
    height = 12;
    width = 15;
    hiddenLayerG = [22, 20, 16];
    hiddenLayerL = [230, 200, 160];
    hiddenLayerC = [180, 150, 100];
    StartExperiment(aeEpoch, nnEpoch, topol,...
        numOfLayer, height, width,...
        hiddenLayerG, hiddenLayerL, hiddenLayerC, keys(i+3));
end

for i = 1:3
    %aeEpoch = 6000;
    %nnEpoch = 4000;
    %topol = 'hextop';
    numOfLayer = i;
    height = 12;
    width = 15;
    hiddenLayerG = [24, 12, 6];
    hiddenLayerL = [240, 120, 60];
    hiddenLayerC = [80, 40, 20];

    StartExperiment(aeEpoch, nnEpoch, topol,...
        numOfLayer, height, width,...
        hiddenLayerG, hiddenLayerL, hiddenLayerC, keys(i+6));
end