fprintf("\nPod-zadatak 6 - Odredjivanje izvjesnosti osmatranja za skraceni niz i najizvjesniji pojedinacni putevi stanja\n");

% Ispis osmatranja prva cetiri izlazna simbola
fprintf('Izvjesnost osmatranja prva cetira izlazna simbola iz prvog niza %e\n', ...
    alpha1_sum(4));

% Prvih 4 osmotrena izlazna simbola iz prvog niza - skraceni niz
data1_1to4 = [3 3 3 4];

% Brojanje slučajeva (broj ćelija u data1)
data1_1to4 = num2cell(data1_1to4, 2);
ncases1to4 = length(data1_1to4);

% Petlja koja obrađuje niz
for m=1:ncases1to4
    % Matrica koja sadrži vjerojatnosti osmatranja skracenog niza u
    % svim stanjima modela
    obslik_1to4 = multinomial_prob(data1_1to4{m}, obsmat0);
    % Izračun matrice unaprijednih i unazadnih vjerojatnosti, pomoćne
    % vjerojatnosti gama za re-estimaciju HMM modela te logaritamske
    % izvjesnosti osmatranja niza
    [alpha1to4, beta1to4, gamma1to4, ll1to4] = ...
        fwdback(prior0, transmat0, obslik_1to4, 'scaled', 0);
    if ll1to4==-inf
        errors = [errors m];
    end
end

% Iskorištavanje funkcije za izračun Viterbijevog puta za skraceni niz
vpath1to4 = viterbi_path(prior0, transmat0, obslik_1to4);
% Iskorištavanje funkcije za izračun log-vjerojatnosti za određeni put (u
% ovom slučaju to je skraćeni niz)
[ll1to4_v, p1to4] = dhmm_logprob_path(prior0, transmat0, obslik_1to4, vpath1to4);

nums = 1:3;
% Inicijalizacija svih mogućih puteva odnosno sve permutacije brojeva 1-3 u
% nizu duljine 4
[A, B, C, D] = ndgrid(nums, nums, nums, nums);
% Stavljanje svih permutacija u jednu matricu       
mpath = [A(:) B(:) C(:) D(:)];

llm=zeros(81,1); % Stupac za log-izvjesnosti
for i=1:81
    % Iskorištavanje funkcije za izračun log-vjerojatnosti za određeni put (u
    % ovom slučaju to je svaki mogući put duljine 4)
    [llm(i), p1to4] = dhmm_logprob_path(prior0, transmat0, obslik_1to4, mpath(i,:));
end

% Sortiranje log-izvjesnosti svih puteva duljine 4 od najvećeg do najmanjeg
[sllm,illm]=sort(-llm);
% Izračunavanje udjela postotka osmatranja svakog puta, udio postotka
% svakog sljedećeg puta se nadodaje na udjele prethodnih puteva (kumulativna suma)
putevi_vjerojatnost = cumsum(exp(-sllm))/sum(exp(llm));

% Ispis udjela izvjesnosti Viterbi puta za ovaj skraceni niz u odnosu na
% sve moguće puteve duljine 4
fprintf('Udio izvjesnosti Viterbi puta: %f\n', putevi_vjerojatnost(1));

% Ispis Viterbi puta za prva 4 izlazna simbola prvog niza
fprintf('Viterbi puta za prva 4 izlazna simbola prvog niza: %d,%d,%d,%d.\n', ...
    vpath1to4(1),vpath1to4(2),vpath1to4(3),vpath1to4(4));

% Ispis broja svih mogucih puteva resetka stanja duljine 4
fprintf('Broj svih mogucih puteva resetka stanja duljine 4 je %d\n', length(mpath));

nedozvoljeni=0;
% Petlja za prebrojavanje puteva koji nisu mogući (to su putevi koji nisu
% ciklički, npr. 1->3->2->1)
for i=1:length(llm)
    if llm(i)==-inf
        nedozvoljeni=nedozvoljeni+1;
    end
end

% Ispis broja nedozvoljenih puteva odnosno puteva koji nisu moguci
fprintf('Broj nedozvoljenih puteva odnosno puteva koji nisu moguci je %d\n', nedozvoljeni);

% Ispis udjela izvjesnosti prvih pet najizvjesnih puteva u odnosu na
% sve moguće puteve duljine 4
fprintf('Udio izvjesnosti prvih pet najizvjesnih puteva: %f\n', putevi_vjerojatnost(5));

% Uzimanje samo prvih put najvjerojatnijih puteva
top_five_indices = illm(1:5);
% U matrici svih puteva pronalazak točno tih pet najvjerojatnijih puteva
top5 = mpath(top_five_indices, :);
fprintf('To su putevi:\n')
disp(top5);