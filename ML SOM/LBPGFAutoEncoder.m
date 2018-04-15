function [AE, GPAutoencoder, LBPAutoencoder, Error, EncoderOutput] = ...
    LBPGFAutoEncoder(Gpoints, LBPpoints, epochs, ...
                     hiddenLayerG, hiddenLayerL, ...
                     hiddenLayerC, numOfLayers)
    
    gMse = zeros(1,numOfLayers);
    lbpMse = zeros(1,numOfLayers);
    concatMse = zeros(1,numOfLayers);
    
    gpOut = Gpoints;
    
    for i=1:numOfLayers
        GPAutoencoder = trainAutoencoder(gpOut, hiddenLayerG(i),...
        'MaxEpochs', epochs,...
        'L2WeightRegularization', 0.001, ...
        'SparsityRegularization', 4, ...
        'SparsityProportion', 0.05, ...
        'DecoderTransferFunction','purelin');
        gMse(i) = mse(predict(GPAutoencoder, gpOut)-gpOut);
        gpOut = encode(GPAutoencoder, gpOut);
        if i==1
            ae1 = GPAutoencoder;
        end
        if i==2
            ae2 = GPAutoencoder;
        end
        if i==3
            ae3 = GPAutoencoder;
        end
    end
    if i == 1
        GPAutoencoder = ae1;
    end
    if i == 2
        GPAutoencoder = stack(ae1,ae2);
    end
    if i ==3
        GPAutoencoder = stack(ae1,ae2,ae3);
    end
    
   % GPAutoencoder = train(GPAutoencoder, Gpoints);
   
   if i == 1
       gpOut = encode(GPAutoencoder, Gpoints);
   else
    gpOut = GPAutoencoder(Gpoints);
   end
   
    lbpOut = LBPpoints;
    for i=1:numOfLayers
        LBPAutoencoder = trainAutoencoder(lbpOut, hiddenLayerL(i),...
        'MaxEpochs', epochs,...
        'L2WeightRegularization', 0.001, ...
        'SparsityRegularization', 4, ...
        'SparsityProportion', 0.05, ...
        'DecoderTransferFunction','purelin');
        lbpMse(i) = mse(predict(LBPAutoencoder, lbpOut)-lbpOut);
        lbpOut = encode(LBPAutoencoder, lbpOut);
        if i==1
            ae1 = LBPAutoencoder;
        end
        if i==2
            ae2 = LBPAutoencoder;
        end
        if i==3
            ae3 = LBPAutoencoder;
        end
    end
    if i==1
        LBPAutoencoder = ae1;
    end
    if i ==2
        LBPAutoencoder = stack(ae1,ae2);
    end
    if i ==3
        LBPAutoencoder = stack(ae1,ae2,ae3);
    end
    
    if i == 1
        lbpOut = encode(LBPAutoencoder,LBPpoints);
    else
        lbpOut = LBPAutoencoder(LBPpoints);
    end
    %{
    Gp = encode(GPAutoencoder, Gpoints);
    Lp = encode(LBPAutoencoder, LBPpoints);
    %}    
    concatP = vertcat(gpOut,lbpOut);
    
    Cp = concatP;
    for i=1:numOfLayers
        
        AE = trainAutoencoder(Cp, hiddenLayerC(i),...
        'MaxEpochs', epochs,...
        'L2WeightRegularization', 0.001, ...
        'SparsityRegularization', 4, ...
        'SparsityProportion', 0.05, ...
        'DecoderTransferFunction','purelin');
        concatMse(i) = mse(predict(AE, Cp)-Cp);
        Cp = encode(AE, Cp);
        if i==1
            ae1 = AE;
        end
        if i==2
            ae2 = AE;
        end
        if i==3
            ae3 = AE;
        end
    end
    if i == 1
        AE = ae1;
    end
    if i == 2
        AE = stack(ae1,ae2);
    end
    if i == 3
        AE = stack(ae1,ae2,ae3);
    end
    
    if i == 1
        EncoderOutput = encode(AE,concatP);
    else
        EncoderOutput = AE(concatP);
    end
    Error = [gMse;lbpMse;concatMse];
    
end