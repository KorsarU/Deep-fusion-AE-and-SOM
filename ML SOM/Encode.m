function cPoints = Encode(ae, points, i)
if i == 1
    cPoints = encode(ae, points);
else
    cPoints = ae(points);
end
    
end