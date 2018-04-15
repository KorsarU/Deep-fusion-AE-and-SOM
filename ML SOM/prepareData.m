function state = prepareData(fileName, data, classLabel)
    %fileName = 'ck.tr';
    fileID = fopen(fileName, 'w');
    fopen(fileID);
    formatSpec = '%d:%f';

    trainRange = 1:lenght(data(:,1));
    oneSampleRange = 1:lenght(data(1,:));
    %for i = ckTrain
    for i = trainRange
        fprintf(fileID, '%d ', classLabel);
        %for j = 1:150
        for j = oneSampleRange
            %fprintf(fileID, formatSpec, j, ckdata(j,i));
            fprintf(fileID, formatSpec, j, data(j,i));
        end
        fprintf(fileID, '\n');
    end

    state = fclose(fileID);
end