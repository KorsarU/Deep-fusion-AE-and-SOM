function [Gpoints, Lpoints, TestSetG, TestSetL] = extractPointsFromFolder(path, labels)
    parts = strsplit(pwd, filesep);
    currentFolder = parts{end};
    if ~(strcmp(currentFolder, path))
        cd(path);
    end
    namesLBPF = ["AngryLBPF.csv", "DisgustLBPF.csv", "FearLBPF.csv",...
                 "HappyLBPF.csv", "SadLBPF.csv",...
                 "SurpriseLBPF.csv"];
    namesGF = ["AngryGF.csv", "DisgustGF.csv", "FearGF.csv",...
               "HappyGF.csv", "SadGF.csv", ...
               "SurpriseGF.csv"];

    %Graphical points
    Gpoints = containers.Map;
    %Test set of graphical points 10\% from each expression type
    TestSetG = containers.Map;
    for i = 1:length(namesGF)
        buf = csvread(namesGF(i));
        b_size = size(buf,1);
        b_dec = int32(fix(b_size/10));
        TestSetG(char(labels(i))) = buf(b_size-b_dec+1:b_size,:);
        %pg = vertcat(pg, buf(1:(b_size-b_dec),:));
        %with shuffle
        %Gpoints(char(labels(i))) = myColShuffle(transp((buf(1:(b_size-b_dec),:))));
        %without shuffle
        Gpoints(char(labels(i))) = transp((buf(1:(b_size-b_dec),:)));
    end

    %Transpose set (nn takes samples as columns) and normalize them by 1
    %(value-min)/(max-min)
    %Gpoints = myColShuffle(transp(myNorm(pg)));

    %LBP points and test set of LBP points 10\% from each expression type
    TestSetL = containers.Map;
    Lpoints = containers.Map;
    for i = 1:length(namesLBPF)
        buf = csvread(namesLBPF(i));
        b_size = size(buf,1);
        b_dec = int32(fix(b_size/10));
        TestSetL(char(labels(i))) = buf(b_size-b_dec+1:b_size,:);
        %pl = vertcat(pl, buf(1:(b_size-b_dec),:));
        %w shuffle
        %Lpoints(char(labels(i))) = myColShuffle(transp((buf(1:(b_size-b_dec),:))));
        %W/o shuffle
        Lpoints(char(labels(i))) = transp((buf(1:(b_size-b_dec),:)));
    end

    %Transpose set (nn takes samples as columns) and normalize them by 1
    %(value-min)/(max-min)
    %Lpoints = myColShuffle(transp(myNorm(pl)));
    cd('..\')
end