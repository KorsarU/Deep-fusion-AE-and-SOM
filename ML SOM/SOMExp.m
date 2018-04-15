    % pos1 = gridtop([10 8]);
% subplot(1,3,1);
% plotsom(pos1);
% pos2 = hextop([10 8]);
% subplot(1,3,2);
% plotsom(pos2);
% pos3 = randtop([10 8]);
% subplot(1,3,3);
% plotsom(pos3);

net1 = selforgmap([10 8]);
net2 = selforgmap([10 8]);
net3 = selforgmap([10 8]);

P = csvread('SourceJaffee.csv');
T = csvread('Test.csv');
rand_pos = randperm(length(P)); 

for k = 1:size(P,1)
    rndP(k,:) = P(rand_pos(k),:);
end
% net1.trainParam.epochs = 1000;
% net1 = train(net1, rndP);
% plotsompos(net1)

if ~net2f
    net2.trainParam.epochs = 400;
    net2 = train(net2,transpose(rndP));
    plotsompos(net2)
    net2f = true;
end
res = zeros(1,80);
for t = 1:size(T,1)
    inp = T(t,:);
    r = sim(net2, inp);
    res(1,r) = res + 1;
end
    
Pl = csvread('AngryLBPF.csv');

Pl = [Pl;csvread('DisgustLBPF.csv')];

Pl = [Pl;csvread('FearLBPF.csv')];

Pl = [Pl;csvread('HappyLBPF.csv')];

Pl = [Pl;csvread('NeutralLBPF.csv')];

Pl = [Pl;csvread('SadLBPF.csv')];

Pl = [Pl;csvread('SurprisedLBPF.csv')];

mmiLBPae = trainAutoencoder(Pl);
Plr = predict(mmiLBPae, Pl);
