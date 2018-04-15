function cPoints = DeepFusionAutoencoder(Gpoints, Lpoints, AEG, AEL, AEC, i)
if i == 1
    gp = encode(AEG, Gpoints);
    lp = encode(AEL, Lpoints);
    cp = vertcat(gp,lp);
    if sum(isnan(cp)) >= 1
        throw(MException('nan in cc', 'nan in cc'));
    end
    cPoints = encode(AEC, cp);
else
    gp = AEG(Gpoints);
    lp = AEL(Lpoints);
    cp = vertcat(gp,lp);
    if sum(isnan(cp)) >= 1
        throw(MException('nan in cc', 'nan in cc'));
    end
    cPoints = AEC(cp);
end
    
end