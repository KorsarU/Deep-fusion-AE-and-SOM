function result = CalculateResultForOne(nn, test, label, labels, height,width, somSetRes)
    result = strings(0);
    v1 = vec2ind(nn(test));
    vu = unique(v1);
    for v = 1:length(vu)
        [x,y] = getXY(vu(v),height,width);
        d = CalcDist(somSetRes,[x,y]);
        win = string(CalcWin(d, labels));

        result = (... 
            ...result contains ...
            ...[#of activated neuron, Expression string, predicted expression string, #num of hits]
                [...
                string(vu(v)),...
                string(label),...
                win,...
                string(sum(v1==vu(v)))]);
    end
end
    
