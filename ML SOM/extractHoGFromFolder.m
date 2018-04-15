function [Hpoints, TestSetH] = extractHoGFromFolder(path, labels)
    testSize = 20; %in %%
    parts = strsplit(pwd, filesep);
    currentFolder = parts{end};
    if ~(strcmp(currentFolder, path))
        cd(path);
    end
    namesHOG = ["AngryHOGF.csv", "DisgustHOGF.csv", "FearHOGF.csv",...
                 "HappyHOGF.csv", "SadHOGF.csv",...
                 "SurpriseHOGF.csv"];
    
    %LBP points and test set of LBP points 10\% from each expression type
    TestSetH = containers.Map;
    Hpoints = containers.Map;
    for i = 1:length(namesHOG)
        label = char(labels(i));
        buf = csvread(namesHOG(i));
        b_size = size(buf,1);
        b_dec = int32(fix(b_size/testSize));
        TestSetH(label) = buf(b_size-b_dec+1:b_size,:);
        %pl = vertcat(pl, buf(1:(b_size-b_dec),:));
        %w shuffle
        %Hpoints(label) = myColShuffle(transp((buf(1:(b_size-b_dec),:))));
        %W/o shuffle
        Hpoints(label) = transp((buf(1:(b_size-b_dec),:)));
    end

    %Transpose set (nn takes samples as columns) and normalize them by 1
    %(value-min)/(max-min)
    %Lpoints = myColShuffle(transp(myNorm(pl)));
    cd('..\')
end