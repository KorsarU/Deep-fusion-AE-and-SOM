%{
allGPoints = [];
for i=1:length(k)
    key = char(k(i));
    allGPoints = [allGPoints, myNorm(CkGpoint(key),1), myNorm(StrilingGpoint(key),1), myNorm(JGpoint(key),1), myNorm(MMIGpointSet(key),1)];
end
allLPoints = [];
for i=1:length(k)
    key = char(k(i));
    allLPoints = [allLPoints, myNorm(CkLpoint(key),1), myNorm(StrilingLpoint(key),1), myNorm(JLpoint(key),1), myNorm(MMILpointSet(key),1)];
end
%}

testT = transp(strAr2targetVec(TheJapaneseFemaleFacialExpressionDatabase1(:,2)));
testD = transp(TheJapaneseFemaleFacialExpressionDatabase);
T = transp(strAr2targetVec(MMITar(:,2)));
D = MMI_orig;
dn= stack(ae,ae_second,ae_third,ae_forth,ae_fifth,ae_sixth,ae_final,softnet);
dn = train(dn, D, T);
expr = dn(D);

test_expr = dn(testD);
subplot(2,1); plotconfusion(T,expr); title('MMI');
subplot(2,2); plotconfusion(testT,test_expr); title('Jaffe as an across test dataset');

