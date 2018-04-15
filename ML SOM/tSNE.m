% T-SNE analysis of data
folders = ["Stirling";"ck"; "jaffe"; "mmi"];
labels =  ["angry","disgust","fear","happy","sad","surprise"];
tSg = containers.Map;
tSl = containers.Map;


colorl = [1 0 0; 0 1 0; 0 0 1; 1 0 1 ; 0.5 0.5 0.3; 0 0.3 0];

%Test after AE

aP = containers.Map;
result= struct;
results = containers.Map;
aeParams = struct;
aeParams.Epochs = 6000;
aeParams.LayersG = [15, 10, 6];
aeParams.LayersL = [200, 150, 100];
aeParams.LayersC = [160, 80, 40];
aeParams.LayersS = [200,150, 75];
aeParams.NumOfLayer = 2;

%prepare all DS and AE, format of result structure:
%result.Name = folder; name equals to DS
%result.CAE = ae; Concatenated AE
%result.GAE = gae; Geometrical AE
%result.LAE = lae; LBP AE
%result.SAE = sae; Stacked AS
%result.SourcePoint.G = pointsG; Original geometrical poitns 
%result.SourcePoint.L = pointsL; Original LBP points
%result.SourcePoint.Stacked = pointsS; Original Stacked Points
%result.ResultPoint = out; Output of CAE
path =  "Result\T-SNE\Before AE\All\";
for i = 1:4
    
    folder = char(folders(i));    
    results(folder) = GetAEfrom(folder, labels, aeParams, false);
    
    %here we have three AE, output from AE, and now we can encode other
    %points from different dataset with that AE and calculate its tSNE
end 


%test for combination of datasets
model = GenerateAEFromCombinationOf(results('mmi'), results('Stirling'), false, aeParams);
model = GenerateAEFromCombinationOf(model, results('ck'), true, aeParams);
fig = PlotTsne(model.SourcePoint.Stacked, 0);
title("Before AE " + model.Name); 

pS = model.SourcePoint.Stacked;
tmpS = containers.Map;
for k = 1:length(labels)
    label = char(labels(k)); 
    if aeParams.NumOfLayer == 1
        tmpS(label) = encode(model.SAE, pS(label));
    else
        tmpS(label) = model.SAE(pS(label));
    end
end

fig = PlotTsne(tmpS, 0);
title("After AE " + model.Name);

% Without encoding
if false
    for i = 1:4
        curr = results(char(folders(i)));
        cP = curr.SourcePoint.Stacked;
        for j = 1:4
            sub = results(char(folders(j)));
            if i == j
                fig = PlotTsne(cP, 0);
                title("Stacked " + curr.Name);
                saveas(fig, char(path + folder + "\" + curr.Name + ".jpg"));
            else
                tmpSS = containers.Map;
                pS = sub.SourcePoint.Stacked;
                for k = 1:length(labels)
                    label = char(labels(k)); 
                    tmpSS([label, ':',sub.Name(1)]) = pS(label);
                end

                fig = PlotTsne(cP, tmpSS);
                title("Stacked " + curr.Name + " " + sub.Name);
                saveas(fig, char(path + folder + "\" + curr.Name + " " + sub.Name + ".jpg"));

            end
        end
    end
end
%PLot staced points in T-SNE for one vs all datasets
if false
    path =  "Result\T-SNE\Deep AE\";
for i = 1:length(folders)
    folder = char(folders(i));
    curr = results(folder);
    for j = 1:length(folders)
        
        sub = results(char(folders(j)));
        
        %got points from curr DS
        pS = curr.SourcePoint.Stacked;
        tmpS = containers.Map;
        for k = 1:length(labels)
            label = char(labels(k)); 
            if aeParams.NumOfLayer == 1
                tmpS(label) = encode(curr.SAE, pS(label));
            else
                tmpS(label) = curr.SAE(pS(label));
            end
        end
        
        if i == j
            fig = PlotTsne(tmpS, 0);
            title("Stacked " + curr.Name);
            saveas(fig, char(path + folder + "\" + curr.Name + " " + sub.Name + ".jpg"));
        else
            tmpSS = containers.Map;
            pS = sub.SourcePoint.Stacked;
            
            for k = 1:length(labels)
                label = char(labels(k)); 
                if aeParams.NumOfLayer == 1
                    tmpSS([label, ':',sub.Name(1)]) = encode(curr.SAE, pS(label));
                else
                    tmpSS([label, ':',sub.Name(1)]) = curr.SAE(pS(label));
                end
            end
            
            fig = PlotTsne(tmpS, tmpSS);
            title("Stacked " + curr.Name + " " + sub.Name);
            saveas(fig, char(path + folder + "\" + curr.Name + " " + sub.Name + ".jpg"));
            
        end
        
        
    end
    
end
end

if false
    path =  "Result\T-SNE\After AE\Each DS by Classes\";
    for i = 1:length(folders)
        folder = char(folders(i));
        model = results(folder);

        for j = 1:length(folders)
            pointC = containers.Map;
            internal_folder = char(folders(j));
            sub = results(internal_folder);
            pointG = sub.SourcePoint.G;
            pointL = sub.SourcePoint.L;
            for k = 1:length(labels)
                label = char(labels(k));
                if aeParams.NumOfLayer == 1
                if size(pointG(label),1) ~= aeParams.LayersG
                    pointG(label) = (encode(model.GAE, pointG(label)));

                end
                if size(pointL(label),1) ~= aeParams.LayersL
                    pointL(label) = (encode(model.LAE, pointL(label)));
                end
                    pointC(label) = (encode(model.CAE, vertcat(pointG(label),pointL(label))));
                else
                    disp('in process');
                end
            end
            fig = PlotTsne(pointG);
            title("G " + model.Name + " " + sub.Name);
            saveas(fig, char(path + folder + "\GP\" + model.Name + " " + sub.Name + ".jpg"));
            fig = PlotTsne(pointL);
            title("L " + model.Name + " " + sub.Name);
            saveas(fig, char(path + folder + "\LBP\" + model.Name + " " + sub.Name + ".jpg"));
            fig = PlotTsne(pointC);
            title("C " + model.Name + " " + sub.Name);
            saveas(fig, char(path + folder + "\Cc\" + model.Name + " " + sub.Name + ".jpg"));

        end
    end
end

%For each label each class
if false
    fpg = containers.Map;
    fpl = containers.Map;
    for j = 1:6
        label = char(labels(j));
        pointsG = containers.Map;
        pointsL = containers.Map;
        for i = 1:4
            folder = char(folders(i));
            [g,l,tg,tl] = extractPointsFromFolder(folder, labels);
            pointsG(folder) = [g(label), tg(label)'];
            pointsL(folder) = [l(label), tl(label)'];
        end
        fpg(label) = pointsG;
        fpl(label) = pointsL;
    end

    for i = 1:6
        f1 = figure;
        label = char(labels(i));
        points = fpl(label);
        for j = 1:4
            folder = char(folders(j));
            yG = tsne(points(folder));
            f1 = gscatter(yG(:,1), yG(:,2));
            f1.Color = colorl(j,:);
            hold on;
        end
        title(label);
        legend(folders);

        hold off;
    end
end


%All
if false
for i = 1:4
    folder = char(folders(i));
    [g,l,tg,tl] = extractPointsFromFolder(folder, labels);
    for j = 1:6
        label = char(labels(j));
        g(label) = [g(label), tg(label)'];
        l(label) = [l(label), tl(label)'];
    end
    g = concatSets(g);
    l = concatSets(l);
    tSg(folder) = tsne(g);
    tSl(folder) = tsne(l);
end

cd Result\T-SNE\All\
f1 = figure;
for i = 1:length(folders)
    folder = char(folders(i));
    yG = tSg(folder);
    f1 = gscatter(yG(:,1), yG(:,2));
    f1.Color = colorl(i,:);
    hold on;
end
legend(folders);
legend();
hold off;
saveas(f1, 'GP/All.jpg');

f2 = figure;
for i = 1:length(folders)
    
    folder = char(folders(i));
    yL = tSl(folder);
    f2 = gscatter(yL(:,1), yL(:,2));
    f2.Color = colorl(i,:);
    hold on;
end
legend(folders);
legend()
hold off;
saveas(f2, 'LBP/All.jpg');

cd ../../../
end