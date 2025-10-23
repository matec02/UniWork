fprintf("\nPod-zadatak 4 - Dekodiranje skrivenih stanja pomocu Viterbi algoritma\n");

% Iskorištavanje funkcije za izračun Viterbijevog puta za prvi niz
vpath1 = viterbi_path(prior0, transmat0, obslik1);
% Prikaz cijelog puta
disp(vpath1);
% Ispis određenih vrijednosti potrebnih za pod-zadatak
fprintf('Prvih tri i zadnji tri vremenska koraka Viterbijevog puta %d,%d,%d,%d,%d,%d.\n', ...
    vpath1(1), vpath1(2), vpath1(3), vpath1(39), vpath1(40), vpath1(41));
