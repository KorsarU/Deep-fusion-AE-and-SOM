function result = GenerateAEFromCombinationOf(model1, model2, needAE, aeParams)
    labels = model1.SourcePoint.G.keys;

    result.Name = [model1.Name, ' ', model2.Name];
    pointsG = containers.Map;
    pointsL = containers.Map;
    pointsS = containers.Map;
    
    for j = 1:length(labels)
        label = char(labels(j));
        
        pointsG(label) = [model1.SourcePoint.G(label), model2.SourcePoint.G(label)];
        pointsL(label) = [model1.SourcePoint.L(label), model2.SourcePoint.L(label)];
        pointsS(label) = [pointsG(label); pointsL(label)];
    
    end
    
    
    g = concatSets(pointsG);
    l = concatSets(pointsL);
    s = concatSets(pointsS);
    if needAE
    [ae, gae, lae, ~, out] = ...
    LBPGFAutoEncoder(g, l, aeParams.Epochs, ...
                     aeParams.LayersG, aeParams.LayersL, ...
                     aeParams.LayersC, aeParams.NumOfLayer);
    
    if aeParams.NumOfLayer == 1
        sae = trainAutoencoder(s, aeParams.LayersS(1),...
                                'MaxEpochs', aeParams.Epochs,...
                                'L2WeightRegularization', 0.001, ...
                                'SparsityRegularization', 4, ...
                                'SparsityProportion', 0.05, ...
                                'DecoderTransferFunction','purelin');
    else
        saes = [];
        sout = s;
        for i = 1:aeParams.NumOfLayer
            saes = [saes;trainAutoencoder(sout, aeParams.LayersS(i),...
                                'MaxEpochs', aeParams.Epochs,...
                                'L2WeightRegularization', 0.001, ...
                                'SparsityRegularization', 4, ...
                                'SparsityProportion', 0.05, ...
                                'DecoderTransferFunction','purelin')];
            tmpAE = saes(i);
            sout = encode(tmpAE,sout);
        end
        if aeParams.NumOfLayer == 2
        sae = stack(saes(1),saes(2));
        else 
            if aeParams.NumOfLayer == 3
                sae = stack(saes(1), saes(2), saes(3));
            end
        end
        sae = train(sae, s);
    end
            
        
    result.CAE = ae;
    result.GAE = gae;
    result.LAE = lae;
    result.SAE = sae;
    result.ResultPoint = out;
    end
    result.SourcePoint.G = pointsG;
    result.SourcePoint.L = pointsL;
    result.SourcePoint.Stacked = pointsS; 
end