function data = AddNoise(orData, fctr)
    
    data = zeros(size(orData), 'like', orData);
    if length(fctr) == 1
        noise = fctr*randn(size(orData));
        data = orData+noise;
    else
        lambda = fctr(1);
        v = fctr(2);
        a = fctr(3);
        if a == 1 %Interpolation
            %[0.5 0 1];
            X = orData'; 
            %lambda = 0.5;
            feature_vector_length = size(X(1,:));
            noised_data = zeros(size(X));
            for i = 1:size(X,1)
               c_k = mean(X(knnsearch(X,X(i,:), 'K', 10),:));
               noised_data(i,:) = (c_k-X(i,:))*lambda+X(i,:);
            end
        end
        
        if a == 2 %Extrapolation
            X = orData'; 
            %lambda = 0.5;
            feature_vector_length = size(X(1,:));
            noised_data = zeros(size(X));
            for i = 1:size(X,1)
               c_k = mean(X(knnsearch(X,X(i,:), 'K', 10),:));
               noised_data(i,:) = (X(i,:)-c_k)*lambda+X(i,:);
            end
        end

        if a == 3
           sz = size(orData);
           x = sz(1);
           y = sz(2);
           for i = 1:x
               for j = 1:y
                   data(i,j) = orData(i,j)+(randn(1)*v+lambda);
               end
           end
        end
    end
end