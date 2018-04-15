function orig = normalizeSizeOfSets(orig)
K=orig.keys;
min = 10^10;
for i = 1:length(K)
    id = char(K(i));
    if min>size(orig(id),2)
        min = size(orig(id),2);
    end
end
for i = 1: length(K)
    id = char(K(i));
    buf = orig(id);
    orig(id) = buf(:,1:min);
end
end