function win = CalcWin(dist, labels)
    pretD = sum(abs(dist));
    win = -1;
    for i = 1:length(dist)
        if dist(i) == -1
            continue;
        end
        if pretD > dist(i)
            win = labels(i);
            pretD = dist(i);
        else
            if pretD == dist(i)
                coin = normrnd(0,1);
                if coin > 0
                    win = labels(i);
                    break;
                else
                    break;
                end
            end
        end
    end
end