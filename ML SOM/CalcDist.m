function dist = CalcDist(som, actN)

    
    labels = unique(som(som~=0));
    dist = zeros(1,length(labels))+length(som)+length(som(:,1));
    
    tx = actN(1);
    ty = actN(2);
    for i=1:length(labels)
        l = labels(i);
        if l == 0
            continue;
        end
        for x = 1:length(som(:,1))
            for y = 1:length(som)
                if som(x,y) == 0 || som(x,y) ~= l
                    continue;
                end
                d = sqrt((x-tx)^2+(y-ty)^2);
                if dist(i) > d
                    dist(i) = d;
                end
            end
        end
    end
    dist(dist == length(som)+length(som(:,1))) = -1;
end