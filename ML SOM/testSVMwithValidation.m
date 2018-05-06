function [result, SVM] = testSVMwithValidation(data)
    keys = data.keys;
    trainData = [];
    trainY = [];
    testData = [];
    testY = [];
    result = 0;
    for i = 1:length(keys)
        key = char(keys(i));
        d = data(key);
        
        testData = d.data;
        testY = d.labels;
        for j = 1:length(keys)
            if i == j
                continue
            end
            k = char(keys(j));
            d = data(k);
            trainData = [trainData, d.data];
            trainY = [trainY; d.labels];
        end
        
        model = fitcecoc(trainData', trainY);
        [~,score] = predict(model, testData');
        labAndScore = [testY, score];
        check = zeros([length(testY), 1]);
        for f = 1:length(testY)
            if labAndScore(f,1) < 0 && labAndScore(f,2) < 0
                check(f) = 1; %True negative
            else
                if labAndScore(f,1) > 0 && labAndScore(f,2) > 0
                    check(f) = 2; % True positive
                else
                    if labAndScore(f,1) > 0 && labAndScore(f,2) < 0
                        check(f) = -1; %False negative
                    else
                        if labAndScore(f,1) < 0 && labAndScore(f,2) > 0
                            check(f) = -2; %False positive
                        end
                    end
                end
            end
        end
        TP = length(check(check==2));
        TN = length(check(check==1));
        FP = length(check(check==-2));
        FN = length(check(check==-1));

        rej = length(score(score<0));
        all = length(testY);
        acc = round(((TP+TN)/all)*100,2);
        
        if acc > result
            SVM = model;
            result = acc;
        end
    end
end