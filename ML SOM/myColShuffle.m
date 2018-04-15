function samples = myColShuffle(matrix)
    samples = matrix;
    data = [1:size(matrix,2)]; % your data
    rand_pos = randperm(length(data)); %array of random positions
    % new array with original data randomly distributed 
    for k = 1:length(data)
        samples(:,k) = matrix(:,rand_pos(k));
    end
end