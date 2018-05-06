function model = GetAEfrom(folder, labels, aeParams, needAE)
    typesOfAE = aeParams.TypesOfAE;
    model.Name = folder;
    [g,l,tg,tl] = extractPointsFromFolder(folder, labels);
    [h, th] = extractHoGFromFolder(folder, labels);
    pointsG = containers.Map;
    pointsL = containers.Map;
    pointsS = containers.Map;
    pointsH = containers.Map;
    pointsNoised = struct;
    
    for j = 1:length(labels)
        label = char(labels(j));
        pointsG(label) = myNorm([g(label), tg(label)'],1);
        pointsL(label) = myNorm([l(label), tl(label)'],1);
        pointsH(label) = myNorm([h(label), th(label)'],1);
        pointsS(label) = [pointsG(label); pointsL(label); pointsH(label)];
    end

    g = concatSets(pointsG);
    l = concatSets(pointsL);
    s = concatSets(pointsS);
    h = concatSets(pointsH);
    
    pointsNoised.Original.Interpolated.G = AddNoise(g,[0.5; 0; 1]);
    pointsNoised.Original.Interpolated.L = AddNoise(l,[0.5; 0; 1]);
    pointsNoised.Original.Interpolated.H = AddNoise(h,[0.5; 0; 1]);

    pointsNoised.Original.Extrapolated.G = AddNoise(g,[0.5; 0; 2]);
    pointsNoised.Original.Extrapolated.L = AddNoise(l,[0.5; 0; 2]);
    pointsNoised.Original.Extrapolated.H = AddNoise(h,[0.5; 0; 2]);

    pointsNoised.Original.WhiteNoise.G = AddNoise(g,[0; 1; 3]);
    pointsNoised.Original.WhiteNoise.L = AddNoise(l,[0; 1; 3]);
    pointsNoised.Original.WhiteNoise.H = AddNoise(h,[0; 1; 3]);
        
    if needAE
        
        if typesOfAE(1) == 1 || typesOfAE(3) == 1
            [ae, gae, lae, ~, out] = ...
            LBPGFAutoEncoder(g, l, aeParams.Epochs, ...
                             aeParams.LayersG, aeParams.LayersL, ...
                             aeParams.LayersC, aeParams.NumOfLayer);
            model.UnNoised.CAE_GL = ae;
            model.UnNoised.GAE = gae;
            model.UnNoised.LAE = lae;
            model.ResultPoint.C_GL = out;
            
            pointsNoised.Encoded.Interpolated.G = ...
                AddNoise(...
                Encode(gae, g, aeParams.NumOfLayer),...
                [0.5; 0; 1]);
            pointsNoised.Encoded.Interpolated.L = ...
                AddNoise(...
                Encode(lae, l, aeParams.NumOfLayer),...
                [0.5; 0; 1]);
            pointsNoised.Encoded.Interpolated.C_GL = ...
                AddNoise(...
                out,...
                [0.5; 0; 1]);
            
            [aeInter, gaeInter, laeInter, ~, outInter] = ...
            LBPGFAutoEncoder(...
                pointsNoised.Original.Interpolated.G,...
                pointsNoised.Original.Interpolated.L,...
                aeParams.Epochs, ...
                aeParams.LayersG, aeParams.LayersL, ...
                aeParams.LayersC, aeParams.NumOfLayer);
            
            model.Noised.Interpolated.CAE_GL = aeInter;
            model.Noised.Interpolated.GAE = gaeInter;
            model.Noised.Interpolated.LAE = laeInter;
            model.ResultPoint.Noised.Interpolated.C_GL = outInter;
            
            pointsNoised.Encoded.Extrapolated.G = ...
                AddNoise(...
                Encode(gae, g, aeParams.NumOfLayer),...
                [0.5; 0; 2]);
            pointsNoised.Encoded.Extrapolated.L = ...
                AddNoise(...
                Encode(lae, l, aeParams.NumOfLayer),...
                [0.5; 0; 2]);
            pointsNoised.Encoded.Extrapolated.C_GL = ...
                AddNoise(...
                out,...
                [0.5; 0; 2]);
            
            [aeExtrap, gaeExtrap, laeExtrap, ~, outExtrap] = ...
            LBPGFAutoEncoder(...
                pointsNoised.Original.Extrapolated.G,...
                pointsNoised.Original.Extrapolated.L,...
                aeParams.Epochs, ...
                aeParams.LayersG, aeParams.LayersL, ...
                aeParams.LayersC, aeParams.NumOfLayer);
            
            model.Noised.Extrapolated.CAE_GL = aeExtrap;
            model.Noised.Extrapolated.GAE = gaeExtrap;
            model.Noised.Extrapolated.LAE = laeExtrap;
            model.ResultPoint.Noised.Extrapolated.C_GL = outExtrap;
            
            pointsNoised.Encoded.WhiteNoise.G = ...
                AddNoise(...
                Encode(gae, g, aeParams.NumOfLayer),...
                [0; 1; 3]);
            pointsNoised.Encoded.WhiteNoise.L = ...
                AddNoise(...
                Encode(lae, l, aeParams.NumOfLayer),...
                [0; 1; 3]);
            pointsNoised.Encoded.WhiteNoise.C_GL = ...
                AddNoise(...
                out,...
                [0; 1; 3]);
            
            [aeWN, gaeWN, laeWN, ~, outWN] = ...
            LBPGFAutoEncoder(...
                pointsNoised.Original.WhiteNoise.G,...
                pointsNoised.Original.WhiteNoise.L,...
                aeParams.Epochs, ...
                aeParams.LayersG, aeParams.LayersL, ...
                aeParams.LayersC, aeParams.NumOfLayer);
            
            model.Noised.WhiteNoise.CAE_GL = aeWN;
            model.Noised.WhiteNoise.GAE = gaeWN;
            model.Noised.WhiteNoise.LAE = laeWN;
            model.ResultPoint.Noised.WhiteNoise.C_GL = outWN;
            

        end
        %CreateAE(points, Layers, Epochs, NumOfLayer)
        if typesOfAE(2) == 1
            
            sae = CreateAE(s, aeParams.LayersS, aeParams.Epochs, aeParams.NumOfLayer);
            model.UnNoised.SAE = sae;
        end
        
        if typesOfAE(3) == 1
            hae = CreateAE(h, aeParams.LayersH, aeParams.Epochs, aeParams.NumOfLayer);
            model.UnNoised.HAE = hae;
            hp = Encode(hae, h, aeParams.NumOfLayer);
            
            pointsNoised.Encoded.Interpolated.H = ...
                AddNoise(hp, [0.5; 0; 1]);
            haeInterpolated = CreateAE(...
                pointsNoised.Original.Interpolated.H,...
                aeParams.LayersH, aeParams.Epochs, aeParams.NumOfLayer);
            model.Noised.Interpolated.HAE = haeInterpolated;
            
            pointsNoised.Encoded.Extrapolated.H = ...
                AddNoise(hp, [0.5; 0; 2]);
            haeExtrapolated = CreateAE(...
                pointsNoised.Original.Extrapolated.H,...
                aeParams.LayersH, aeParams.Epochs, aeParams.NumOfLayer);
            model.Noised.Extrapolated.HAE = haeExtrapolated;
            
            pointsNoised.Encoded.WhiteNoise.H = ...
                AddNoise(hp, [0; 1; 3]);
            haeWhiteNoise = CreateAE(...
                pointsNoised.Original.WhiteNoise.H,...
                aeParams.LayersH, aeParams.Epochs, aeParams.NumOfLayer);
            model.Noised.WhiteNoise.HAE = haeWhiteNoise;
            
            CC = containers.Map;
            for j = 1:length(labels)
                expression = char(labels(j));
                hp = pointsH(expression);
                lp = pointsL(expression);
                gp = pointsG(expression);
                sh = size(hp);
                sg = size(gp);
                sl = size(lp);
                sh = sh(2);
                sg = sg(2);
                sl = sl(2);
                if (sl+sg)-2*sh ~= 0
                    if (sl+sg)-2*sh > 0
                        tmp = pointsH(expression);
                        tmp = [tmp, tmp(:,sh)];
                        pointsH(expression) = tmp;
                        hp = tmp;
                    else
                        if sl > sg 
                            fprintf("error in datasize %s\n", folder);
                        end
                    end
                end
                        
                    

                gp = Encode(gae, gp, aeParams.NumOfLayer);
                lp = Encode(lae, lp, aeParams.NumOfLayer);
                hp = Encode(hae, hp, aeParams.NumOfLayer);
                cp = vertcat(vertcat(gp,lp), hp);
                CC(expression) = cp;
                

            
            end
            %!!! wrong!!!
            [trainCC, ~] = concatSets(CC);
            model.UnNoised.CAE_GLH = CreateAE(trainCC, aeParams.LayersC, aeParams.Epochs, aeParams.NumOfLayer);
            model.ResultPoint.C_GLH = Encode(model.UnNoised.CAE_GLH, trainCC, aeParams.NumOfLayer);
            
            pointsNoised.Encoded.Interpolated.C_GLH = ...
                AddNoise(model.ResultPoint.C_GLH, [0.5; 0; 1]);
            pointsNoised.Encoded.Extrapolated.C_GLH = ...
                AddNoise(model.ResultPoint.C_GLH, [0.5; 0; 2]);
            pointsNoised.Encoded.WhiteNoise.C_GLH = ...
                AddNoise(model.ResultPoint.C_GLH, [0; 1; 3]);
            
            trainCC = vertcat(vertcat(...
                pointsNoised.Encoded.Interpolated.G,...
                pointsNoised.Encoded.Interpolated.L),...
                pointsNoised.Encoded.Interpolated.H);
            model.Noised.Interpolated.CAE_GLH = CreateAE(trainCC,...
                aeParams.LayersC, aeParams.Epochs, aeParams.NumOfLayer);
            
            trainCC = vertcat(vertcat(...
                pointsNoised.Encoded.Extrapolated.G,...
                pointsNoised.Encoded.Extrapolated.L),...
                pointsNoised.Encoded.Extrapolated.H);
            model.Noised.Extrapolated.CAE_GLH = CreateAE(trainCC,...
                aeParams.LayersC, aeParams.Epochs, aeParams.NumOfLayer);
            
            trainCC = vertcat(vertcat(...
                pointsNoised.Encoded.WhiteNoise.G,...
                pointsNoised.Encoded.WhiteNoise.L),...
                pointsNoised.Encoded.WhiteNoise.H);
            model.Noised.WhiteNoise.CAE_GLH = CreateAE(trainCC,...
                aeParams.LayersC, aeParams.Epochs, aeParams.NumOfLayer);
        end
        
    end
    model.SourcePoint.TG = tg;
    model.SourcePoint.TL = tl;
    model.SourcePoint.TH = th;
    model.SourcePoint.G = pointsG;
    model.SourcePoint.L = pointsL;
    model.SourcePoint.H = pointsH;
    model.SourcePoint.Stacked = pointsS;
    model.aeParams = aeParams;
    model.NoisedData = pointsNoised;
    
    
    
end