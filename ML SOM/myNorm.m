function normMatr = myNorm(mat, dim)
    normMatr = mat;

    if dim == 2
        for i = 1:size(mat,2)
            num = (mat(:,i)-min(mat(:,i)));
            denum = (max(mat(:,i))-min(mat(:,i)));
            if denum == 0
                if max(mat(:,i)) == 0
                    normMatr(:,i) = 0;
                else if (max(mat(:,i))-min(mat(:,i))) == 0
                    normMatr(:,i) = 1;
                    end
                end
                continue;
            end
            normMatr(:,i) = num/denum;
        end
    else
        for i = 1:size(mat,1)
            num = (mat(i,:)-min(mat(i,:)));
            denum = (max(mat(i,:))-min(mat(i,:)));
            if denum == 0
                if max(mat(i,:)) == 0
                    normMatr(i,:) = 0;
                else if (max(mat(i,:))-min(mat(i,:))) == 0
                    normMatr(i,:) = 1;
                    end
                end
                continue;
            end
            normMatr(i,:) = num/denum;
        end
end