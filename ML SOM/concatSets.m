function [trainingData, trainLabels] = concatSets(set)
    trainingData = [];
    trainLabels = strings(0);
    K = set.keys;
    for i = 1:length(K)
        data = set(char(K(i)));
        trainingData = horzcat(trainingData, data);
        for j = 1:length(data(1,:))
            trainLabels = horzcat(trainLabels, char(K(i)));
        end
    end
end