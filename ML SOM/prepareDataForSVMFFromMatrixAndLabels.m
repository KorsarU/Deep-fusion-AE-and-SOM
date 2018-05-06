function res = prepareDataForSVMFFromMatrixAndLabels(data, labels, folds_num)
    classSizes = zeros(size(unique(labels)));
    unLabels = unique(labels);
    for i = 1:length(classSizes)
        classSizes(i) = sum(labels == unLabels(i));
    end
    assert(sum(classSizes)==size(data,2));
    
    folds = containers.Map;
    
    %split data on labels
    dataPerLabels = containers.Map;
    s = 0;
    for l = 1:length(unLabels)
        label = char(unLabels(l));
        dataPerLabels(label) = data(:,s+1:s+classSizes(l));
        s = s+classSizes(l);
    end
    assert(size(data,2)==s);
    
    %for each fold
    folds = containers.Map;
    for l = 1:length(unLabels)
        label = char(unLabels(l));
        d = dataPerLabels(label);
        sd = classSizes(l);
        step = round(sd/folds_num);
        class_folds = containers.Map;
        checksum = 0;
        for f = 1:folds_num
            if f < folds_num
                class_folds(num2str(f)) = d(:,step*(f-1)+1:step*f);
                
            else
                if f == folds_num
                    class_folds(num2str(f))=...
                        d(:,step*(f-1)+1:classSizes(l));
                    
                end
            end
            checksum = checksum + size(class_folds(num2str(f)),2);
        end
        folds(label) = class_folds;
        if checksum ~= classSizes(l)
            fprintf('mismatch checksum (checksum/label/class size): %d, %s, %d\n', checksum, label, classSizes(l));
        end
    end
    
    res = containers.Map;
    s = 0;
    for f = 1:folds_num
        kfolds = struct;
        fold_data = [];
        fold_labels = [];
        for l = 1:length(unLabels)
            label = char(unLabels(l));
            data_buf = folds(label);
            data_buf = data_buf(num2str(f));
            fold_data = [fold_data, data_buf];
            
            label_buf = strings(1,size(data_buf,2));
            label_buf(:) = string(label);
            s = s+size(data_buf,2);
            fold_labels = [fold_labels, label_buf];
        end
        kfolds.data = fold_data;
        fold_labels = LabelsToIndex(fold_labels);
        kfolds.labels = fold_labels;
        res(num2str(f)) = kfolds;
    end
    assert(s==size(data,2));    
end