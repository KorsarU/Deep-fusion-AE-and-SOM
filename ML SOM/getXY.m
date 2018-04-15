function [h,v] = getXY(n, height, width)
    h = height-fix(n/width);
    if h == 0
        h = height;
    end
    v = mod(n,width);
    if v == 0
        v = width;
    end
end