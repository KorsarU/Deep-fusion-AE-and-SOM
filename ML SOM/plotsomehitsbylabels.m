function resulted = plotsomehitsbylabels(nn, data, labels, height, width, draw, tt)
    hits = containers.Map();
    
    dataPerLabels = containers.Map();
    l = unique(labels);
        
    if isa(data, 'containers.Map')
        dataPerLabels = data;
    else
        for i = 1:length(l)
            dataPerLabels(char(l(i))) = [];
        end
        for i = 1:length(data(1,:))
            key = char(labels(i));
            tmp = horzcat(dataPerLabels(key), data(:,i));
            dataPerLabels(key) = tmp;
        end
    end
    for i = 1:length(l)
        key = char(l(i));
        hits(key) = vec2ind(nn(dataPerLabels(key)));
    end

    resulted = zeros(height, width);
    an = zeros(height, width);
    logA = containers.Map();
    for k = 1:length(hits.keys)
        a = zeros(height, width);
        lA = zeros(height,width);
        vL = hits(char(l(k)));
        for i = 1:length(vL)
            [h, v] = getXY(vL(i), height, width);
            a(h, v) = a(h, v)+1;

        end

        logA(char(l(k))) = a;
        vU = unique(vL);
        for i = 1:length(vU)

            [h, v] = getXY(vU(i), height, width);
            neuronCost = sumofneibors(a,h,v);
            rule = a(h,v) == 1;
            if rule
                continue;
            end
            if an(h,v) < neuronCost
                an(h,v) = neuronCost;
                resulted(h,v) = 0;
            else
                if an(h,v) == neuronCost
                    %Simple head&tails play p = 0.5
                    while(true)
                        coin = normrnd(0,1);
                        if coin > 0
                            a(h,v) = 0;
                            break
                        else
                            if coin < 0
                                resulted(h,v) = 0;
                                break
                            end
                        end
                    end
                else
                    a(h,v) = 0;
                end
            end
        end
        
        %things to check result som
        %if we lost some previous result - test is failed
        %how it works
        %we have k, in result matrix the biggest number is k-1
        %the problem is in intersection, when we have k-1+k i.e. 2k-1
        %to check that, we calculated sum of all k-1
        %after adding new result we check changes in sum of k-1
        %it has to be simular
        %if we have situation, than k+k-n, where k-n is result for
        %preprevious measurement, we can check it by looking results which
        %are more than k
        checkRule = sum(resulted(resulted==k-1));
        %disp(l(k));
        lA(a>1) = k;
        resulted = resulted+lA;
        if checkRule ~= sum(resulted(resulted==k-1)) ||...
                sum(resulted(resulted>k))>0
            disp('panic')
            disp(getXY(resulted(resulted(resulted>=k+1))))
        end


    end

    if draw>0
        C = [1 1 1;...
            1 0 0;...
            1 0.5 0;...
            1 1 0;...
            0 1 0;...
            0 0.5 1;...
            0 0 1];
        drawColoredSOM(resulted, l, height, width, C, 1, char(tt));
    end

end

