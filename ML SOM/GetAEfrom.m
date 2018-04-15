function result = GetAEfrom(folder, labels, aeParams, needAE)
    typesOfAE = aeParams.TypesOfAE;
    result.Name = folder;
    [g,l,tg,tl] = extractPointsFromFolder(folder, labels);
    [h, th] = extractHoGFromFolder(folder, labels);
    pointsG = containers.Map;
    pointsL = containers.Map;
    pointsS = containers.Map;
    pointsH = containers.Map;
    for j = 1:length(labels)
        label = char(labels(j));
        pointsG(label) = myNorm([g(label), tg(label)'],1);
        pointsL(label) = myNorm([l(label), tl(label)'],1);
        pointsS(label) = [pointsG(label); pointsL(label)];
        pointsH(label) = myNorm([h(label), th(label)'],1);
    end
    
    g = concatSets(pointsG);
    l = concatSets(pointsL);
    s = concatSets(pointsS);
    h = concatSets(pointsH);
    if needAE
        
        if typesOfAE(1) == 1
            [ae, gae, lae, ~, out] = ...
            LBPGFAutoEncoder(g, l, aeParams.Epochs, ...
                             aeParams.LayersG, aeParams.LayersL, ...
                             aeParams.LayersC, aeParams.NumOfLayer);
            result.CAE = ae;
            result.GAE = gae;
            result.LAE = lae;
            result.ResultPoint = out;
        end
        %CreateAE(points, Layers, Epochs, NumOfLayer)
        if typesOfAE(2) == 1
            
            sae = CreateAE(s, aeParams.LayersS, aeParams.Epochs, aeParams.NumOfLayer);
            result.SAE = sae;
        end
        
        if typesOfAE(3) == 1
            hae = CreateAE(h, aeParams.LayersH, aeParams.Epochs/6, aeParams.NumOfLayer);
            result.HAE = hae;
        end
        
    end
    result.SourcePoint.TG = tg;
    result.SourcePoint.TL = tl;
    result.SourcePoint.TH = th;
    result.SourcePoint.G = pointsG;
    result.SourcePoint.L = pointsL;
    result.SourcePoint.H = pointsH;
    result.SourcePoint.Stacked = pointsS;
    result.aeParams = aeParams;
    
    
    
end