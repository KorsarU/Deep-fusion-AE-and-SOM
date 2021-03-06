function targetArray = strAr2targetVec(strArray)
    target = string(strArray);
    T = zeros(length(target),6);
    for i=1:length(target)
        x = upper(target(i));
        if x == "ANGRY"
            j = 1;
        end
        if x ==  "DISGUST"
            j = 2;
        end
        if x ==  "FEAR"
            j = 3;
        end
        if x == "HAPPY"
            j = 4;
        end
        if x == "SAD"
            j = 5;
        end
        if x == "SURPRISE"
            j = 6;
        end
        T(i,j) = 1;
    end
    targetArray = T;
end