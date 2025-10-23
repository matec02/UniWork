fprintf("\nPod-zadatak 7 - Generiranje opservacija za zadani model\n");

rng('default');
% Generiranje visestrukog opservacijskog niza:
T = 125; % duljina svakog niza
nex = 14; % broj opservacijskih nizova
% Inicijalizacija matrice koja sadrži 14 nasumično generiranih nizova duljine 125
data = dhmm_sample(prior0, transmat0, obsmat0, nex, T);
% Ispis matrice
disp(data);