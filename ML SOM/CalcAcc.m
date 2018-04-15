
function ac = CalcAcc(test)

    overallAcc = 0;
    labels = [];
    for i = 1:length(test)
        labels = horzcat(labels, test(i,2), test(i,3));
    end
    labels = unique(labels);
    
    true_pos = containers.Map;
    true_neg = containers.Map;
    false_pos = containers.Map;
    false_neg = containers.Map;
    all = containers.Map;
    all_sum = 0;
    for l = labels
        l = char(l);
        true_neg(l) = 0;
        true_pos(l) = 0;
        false_neg(l) = 0;
        false_pos(l) = 0;
        all(l) = 0;

    end
    for i = 1:length(test)
        exp = char(test(i,2));
        pre = char(test(i,3));
        num = test(i,4);
        s   = str2double(num);
        if strcmp(exp, pre)
            true_pos(exp) = true_pos(exp)+s;
        else
            false_neg(exp) = false_neg(exp)+s;
        end
            all(exp) = all(exp)+s;
        all_sum = all_sum+s;
    end
    for i = 1:length(test)
        exp = char(test(i,2));
        pre = char(test(i,3));
        num = test(i,4);
        s   = str2double(num);
        if ~strcmp(exp, pre)
            false_pos(pre) = false_pos(pre)+s;
        end
    end
    for i = labels
        l = char(i);
        true_neg(l) = all_sum - true_pos(l)-false_pos(l)-false_neg(l);
    end
    for i = labels
        l = char(i);
        ...disp(l);
        ...disp("all = "+all(l));
        ...disp("TP = "+true_pos(l));
        ...disp("TN = "+true_neg(l));
        ...disp("FP = "+false_pos(l));
        ...disp("FN = "+false_neg(l));
        ac = (true_neg(l)+true_pos(l)*1.00)/(true_neg(l)+true_pos(l)...
            +false_neg(l)+false_pos(l));
        ac = ac*100;
        ...disp("AC = "+ac+"%");
        overallAcc = overallAcc+ac;
    end
    ac = overallAcc/(length(labels));
end