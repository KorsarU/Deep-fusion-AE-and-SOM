function result = CalculateResult(nn, testSet, labels, height,width, somSetRes)
    result = strings(0);
    for j = 1:length(labels)
        exT = char(labels(j));
        %disp(exT);
 
        v1 = vec2ind(nn(testSet(exT)));
        vu = unique(v1);
        for v = 1:length(vu)
            [x,y] = getXY(vu(v),height,width);
            d = CalcDist(somSetRes,[x,y]);
            win = string(CalcWin(d, labels));
            
            result = vertcat(... 
                ...result contains ...
                ...[#of activated neuron, Expression string, predicted expression string, #num of hits]
                result, ...
                [...
                string(vu(v)),...
                string(exT),...
                win,...
                string(sum(v1==vu(v)))]);
        end
    end
    
end