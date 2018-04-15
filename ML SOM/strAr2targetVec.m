function targetArray = strAr2targetVec(strArray)
    target = string(strArray);
    T = zeros(length(target),6);
    for i=1:length(target)
        x = target(i);
        if x == "AN"
           j = 1;
        end
        if x ==  "DI"
          j = 2;
        end
        if x ==  "FE"
          j = 3;
        end
        if x == "HA"
          j = 4;
        end
        if x == "SA"
           j = 5;
        end
        if x == "SU"
           j = 6;
        end
        T(i,j) = 1;
    end
    targetArray = T;
end