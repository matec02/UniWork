fprintf("\nPod-zadatak 5 - Odredjivanje log-izvjesnosti osmatranja uzduz dekodiranih Viterbi puteva\n");

% Iskorištavanje funkcije za izračun Viterbijevog puta za drugi niz
vpath2 = viterbi_path(prior0, transmat0, obslik2);

% Iskorištavanje funkcije za izračun log-vjerojatnosti za određeni put (u
% ovom primjeru to su Viterbijevi putevi za prvi i drugi niz)
[ll1_v, p1_v] = dhmm_logprob_path(prior0, transmat0, obslik1, vpath1);
[ll2_v, p2_v] = dhmm_logprob_path(prior0, transmat0, obslik2, vpath2);

% Ispis log-izvjesnosti za Viterbi puteve
fprintf('Log-izvjesnost za Viterbi put prvog niza: %f\n', ll1_v);
fprintf('Log-izvjesnost za Viterbi put drugog niza: %f\n', ll2_v);

% Ispis razlike log izvjesnosti preko svih puteva i uzduz Viterbi puta za
% određeni niz
fprintf('Razlika log-izvjesnosti preko svih puteva i uzduz Viterbi puta za prvi niz: %f\n', ...
    ll1-ll1_v);
fprintf('Razlika log-izvjesnosti preko svih puteva i uzduz Viterbi puta za drugi niz: %f\n', ...
    ll2-ll2_v);