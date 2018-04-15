function data = AddNoise(orData, fctr)
    noise = fctr*randn(size(orData));
    data = orData+noise;
end