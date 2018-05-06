function cPoints = DeepFusionAutoencoderPlusHOG(Gpoints, Lpoints, Hpoints, AEG, AEL, AEH, AEC, i)
if i == 1
    gp = encode(AEG, Gpoints);
    lp = encode(AEL, Lpoints);
    hp = encode(AEH, Hpoints);
    cp = vertcat(vertcat(gp,lp), hp);
    if sum(isnan(cp)) >= 1
        throw(MException('nan in cc', 'nan in cc'));
    end
    cPoints = encode(AEC, cp);
else
    gp = AEG(Gpoints);
    lp = AEL(Lpoints);
    hp = AEH(Hpoints);
    cp = vertcat(vertcat(gp,lp), hp);
    if sum(isnan(cp)) >= 1
        throw(MException('nan in cc', 'nan in cc'));
    end
    cPoints = AEC(cp);
end
    
end