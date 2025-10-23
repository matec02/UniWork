fprintf("Podzadatak 1 - Cjelovito definiranje HMM modela u Matlabu\n");
clear;
% Dodavanje puta na biblioteku funkcija
addpath(genpath('C:\Users\Matija Jakovac\Documents\MATLAB\HMMall')) 

% Vektor inicijalne vjerojatnosti stanja

prior0=[
    1 % Prva kocka (ako je palo '1')
    2 % Druga kocka (ako je palo '2' ili '3')
    3 % Treca kocka (ako je palo '4', '5' ili '6')
]/6;

% Broj stanja HMM modela
Q=size(prior0,1);

M=9; % personalizirani M

% Matrica vjerojatnosti promjena stanja
%
% a11 a12 a13
% a21 a22 a23
% a31 a32 a33

% Formiranje matrice vjerojatnosti prijelaza stanja
% (uz ciklicku strukturu izmjene stanja, jer su
% prijelazi 1->3, 2->1 i 3->2 zabranjeni)
transmat0 = [
M-1 1 0 % P(1|1) P(2|1) P(3|1)
0 M-1 1 % P(1|2) P(2|2) P(3|2)
1 0 M-1 % P(1|3) P(2|3) P(3|3)
]/M;

% Matrica emisijskih vjerojatnosti
% svaki redak odgovara jednom stanju, a
% svaki stupac jednoj mogucoj opservaciji
obsmat0 = [
20 3 1 6 4 6  % P(1|1. kocka) P(2|1. kocka) P(3|1. kocka) P(4|1. kocka) P(5|1. kocka) P(6|1. kocka)
4 4 20 2 3 7  % P(1|2. kocka) P(2|2. kocka) P(3|2. kocka) P(4|2. kocka) P(5|2. kocka) P(6|2. kocka)
2 5 5 6 20 2  % P(1|3. kocka) P(2|3. kocka) P(3|3. kocka) P(4|3. kocka) P(5|3. kocka) P(6|3. kocka)
]/40;

% Ukupni broj simbola rječnika - u ovom slučaju 6 (1,2,3,4,5 ili 6 dobiveno na kocki)
O=size(obsmat0, 2); 

fprintf("\nPod-zadatak 2 - Odredjivanje log-izvjesnosti osmatranja zadanog izlaznog niza simbola za zadani model\n");

% Inicijalizacija podataka
data1 = [3 3 3 4 4 5 5 5 3 4 1 1 1 1 1 4 1 2 4 1 4 5 1 6 1 1 1 6 4 1 2 6 6 1 3 3 3 6 3 1 1;
         2 2 6 4 6 2 2 2 6 4 2 5 1 2 6 5 6 6 6 1 2 5 4 1 1 4 6 1 3 6 5 6 6 6 1 6 6 1 3 1 6];

% Pretvaranje matrice u ćelije, svaki red postaje zasebna ćelija
data1 = num2cell(data1, 2);

% Brojanje slučajeva (broj ćelija u data1)
ncases1 = length(data1);

% Inicijalizacija niza za zabilježavanje greški
errors = [];

% Petlja koja obrađuje pojedinačni niz
for m = 1:ncases1
    % Izračunavanje za prvi niz 
    if m == 1
        % Matrica koja sadrži vjerojatnosti osmatranja prvog niza u
        % svim stanjima modela
        obslik1 = multinomial_prob(data1{m}, obsmat0);
        % Izračun matrice unaprijednih i unazadnih vjerojatnosti, pomoćne
        % vjerojatnosti gama za re-estimaciju HMM modela te logaritamske
        % izvjesnosti osmatranja niza
        [alpha1, beta1, gamma1, ll1] = ...
            fwdback(prior0, transmat0, obslik1, 'scaled', 0);
        fprintf('Log izvjesnost prvog niza je %f\n', ll1);
        if ll1 == -inf
            errors = [errors m];
        end
    % Izračunavanje za drugi niz
    elseif m == 2
        % Matrica koja sadrži vjerojatnosti osmatranja drugog niza u
        % svim stanjima modela
        obslik2 = multinomial_prob(data1{m}, obsmat0);
        % Izračun matrice unaprijednih i unazadnih vjerojatnosti, pomoćne
        % vjerojatnosti gama za re-estimaciju HMM modela te logaritamske
        % izvjesnosti osmatranja niza
        [alpha2, beta2, gamma2, ll2] = ...
            fwdback(prior0, transmat0, obslik2, 'scaled', 0);
        fprintf('Log izvjesnost drugog niza je %f\n', ll2);
        if ll2 == -inf
            errors = [errors m];
        end
    end
end

% Sumiranje vrijednosti u matricama alpha1 i alpha2
% kako bi dobili ukupnu vjerojatnost osmatranja niza
alpha1_sum = sum(alpha1);
alpha2_sum = sum(alpha2);

% Izračunavanje i ispis koliko je drugi niz manje vjerojatan nego li prvi
% niz
difference = alpha1_sum(41) / alpha2_sum(41);
fprintf('Drugi niz je %e manje vjerojatan nego prvi\n', difference);

fprintf("\nPod-zadatak 3 - Izracunavanje vjerojatnosti unaprijed i unazad za sva skrivena stanja modela i sve vremenske trenutke osmatranja\n");

% Prikaz unaprijedne vjerojatnosti za prvu sekvencu pomoću točnog elementa
% matrice odnosno vjerojatnosti da je u vremenskom trenutku 27 osmotreno 
% stanje 1
fprintf('Alpha u vremenskom trenutku 27 za prvo stanje iznosi %e\n', alpha1(1,27))
% Prikaz unazadne vjerojatnosti za prvu sekvencu pomoću točnog elementa
% matrice odnosno vjerojatnosti da je u vremenskom trenutku 12 osmotreno 
% stanje 2
fprintf('Beta u vremenskom trenutku 12 za drugo stanje iznosi %e\n', beta1(2,12))


fprintf('Ukupna vjerojatnost osmatranja prvog niza dobivena preko alfe %e\n', alpha1_sum(41))

% Umnožak i sumacija prvog vremenskog koraka bete, vektora inicijalne
% matrice te vektor izlazne vjerojatnosti konkretnog osmotrenog simbola u
% prvom vremenskom koraku
beta_ukupna_vjerojatnost = sum(beta1(:,1).*prior0.*obslik1(:,1));
fprintf('Ukupna vjerojatnost osmatranja prvog niza dobivena preko bete %e\n', beta_ukupna_vjerojatnost);

fprintf("\nPod-zadatak 4 - Dekodiranje skrivenih stanja pomocu Viterbi algoritma\n");

% Iskorištavanje funkcije za izračun Viterbijevog puta za prvi niz
vpath1 = viterbi_path(prior0, transmat0, obslik1);
% Prikaz cijelog puta
disp(vpath1);
% Ispis određenih vrijednosti potrebnih za pod-zadatak
fprintf('Prvih tri i zadnji tri vremenska koraka Viterbijevog puta %d,%d,%d,%d,%d,%d.\n', ...
    vpath1(1), vpath1(2), vpath1(3), vpath1(39), vpath1(40), vpath1(41));

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

fprintf("\nPod-zadatak 7 - Generiranje opservacija za zadani model\n");

rng('default');
% Generiranje visestrukog opservacijskog niza:
T = 125; % duljina svakog niza
nex = 14; % broj opservacijskih nizova
% Inicijalizacija matrice koja sadrži 14 nasumično generiranih nizova duljine 125
data = dhmm_sample(prior0, transmat0, obsmat0, nex, T);
% Ispis matrice
disp(data);

fprintf("\nPod-zadatak 8 - Odredjivanje dugotrajne statistike osmotrenih simbola i usporedba s njihovim teorijskim ocekivanjima\n");

% Prebrojavanje simbola u svakoj sekvenci
hm=hist(data',[1 2 3 4 5 6]);

% Ispis osmatranja izlaznih simbola za prvu sekvencu
fprintf('Broj osmatranja izlaznih simbola za prvu sekvencu, redom od 1 do 6: \n');
disp(hm(:,1));

% Određivanje stacionarnu distribuciju stanja (pi_stac) uzastopnim mnozenjem 
% zadane prijelazne matrice A same sa sobom i to T puta
pi_stac=transmat0; for i=1:T, pi_stac=pi_stac*transmat0; end;

% Ispis dugotrajne vjerojatnosti stanja modela 1
fprintf('Dugotrajna vjerojatnost stanje modela 1 je %f\n', pi_stac(1,1))

% Određivanje dugotrajne statistike izlaznih simbola odnosno njihove
% vjerojatnosti da se dogode kroz dulje vrijeme
dugotrajna_statistika = pi_stac(1,:)*obsmat0;

% Ispis dugotrajne vjerojatnosti osmatranja izlaznog simbola 4
fprintf('Dugotrajna vjerojatnost osmatranja izlaznog simbola 4 je %f\n', dugotrajna_statistika(4));

% Empirijske vjerojatnosti koje smo dobili prebrojavanje simbola odnosno
% njihova vjerojatnost da se dogode u našim nasumično odabranim slučajevima
empirijska = mean(hm')/T;

% Izračun razlike između statistike i empirijskih vjerojatnost za svaki
% izlazni simbol
razlike = abs(dugotrajna_statistika - empirijska);

% Maksimalna razlika između statistike i empirijskih vjerojatnost za neki
% od izlaznih simbola
max_razlika = max(razlike);

% Ispis maksimalne razlike između statistike i empirijskih vjerojatnost za neki
% od izlaznih simbola
fprintf(['Maksimalna razlike između statistike i empirijskih vjerojatnost za ' ...
    'neki od izlaznih simbola je %f\n'], max_razlika)

fprintf("\nPod-zadatak 9 - Izracun log-izvjesnosti osmatranja pojedinacnih generiranih opservacija temeljem zadanog modela\n");

% Inicijalizacija polja u kojemu će biti pohranjene log-izvjesnost svih 14
% nasumično odabranih nizova
ll_data = [];

for i=1:nex
    for m=1:1
        % Uzimanje pojedinačnog niza 
        data_row = data(i, :);
        % Pretvaranje niza u ćeliju
        data_row = num2cell(data_row, 2);
        % Matrica koja sadrži vjerojatnosti osmatranja pojedinačnog niza u
        % svim stanjima modela
        obslik = multinomial_prob(data_row{m}, obsmat0);
        % Izračun matrice unaprijednih i unazadnih vjerojatnosti, pomoćne
        % vjerojatnosti gama za re-estimaciju HMM modela te logaritamske
        % izvjesnosti osmatranja niza
        [alpha, beta, gamma, ll] = ...
            fwdback(prior0, transmat0, obslik, 'scaled', 0);
        % Dodavanje pojedinačnih log-izvjesnosti u polje
        ll_data(end+1) = ll;
        if ll==-inf
            errors = [errors m];
        end
    end
end

% Pronalazak srednje, minimalne i maksimalne vrijednost za log-izvjesnost
% 14 nasumičnih nizova
sr_vr_data = mean(ll_data);
min_data = min(ll_data);
max_data = max(ll_data);

% Ispis navedenih vrijednosti
fprintf(['Maksimalna vrijednost: %f\nMinimalna vrijednost: %f\nSrednja vrijednost: ' ...
    '%f\n'], max_data, min_data, sr_vr_data);

fprintf("\nPod-zadatak 10 - Provedite postupak treniranja parametara HMM modela\n");

% Resetiranje generatora pseudo-slucajnih brojeva na pocetnu vrijednost
rng('default');

% Inicijalizacija potpuno slučajnih parametara modela
% Iskorišteni varijable Q - broj stanja i O - broj izlaznih modela
prior1 = normalise(rand(Q,1)); % Matrica početnih vjerojatnosti
transmat1 = mk_stochastic(rand(Q,Q)); % Matrica prijelaznih vjerojatnosti
obsmat1 = mk_stochastic(rand(Q,O)); % Matrica emisijskih vjerojatnosti

% Učenje parametara modela temeljeno na algoritmu maksimizacije očekivanja
% za model s potpuno slučajnih parametrima
% Ispis iteracija za prvi HMM model
fprintf('Ispis iteracija za prvi HMM model sa slučajnim parametrima:\n')
[LL2, prior2, transmat2, obsmat2] = dhmm_em(data, prior1, transmat1, obsmat1, 'max_iter', 200, 'thresh', 1E-6);

% Ispis iteracija za prvi HMM model
fprintf('\nIspis iteracija za drugi HMM model sa već zadanim parametrima:\n')

% Učenje parametara modela temeljeno na algoritmu maksimizacije očekivanja
% za model s već zadanim parametrima
[LL3, prior3, transmat3, obsmat3] = dhmm_em(data, prior0, transmat0, obsmat0, 'max_iter', 200, 'thresh', 1E-6);

% Za oba učenja je broj iteracija EM postupka na najvise 200, a prag 
% relativne promjene izvjesnosti u odnosu na proslu iteraciju za zavrsetak 
% postupka je na 1E-6

fprintf("\nPod-zadatak 11 - Usporedna evaluacija zadanog modela, slucajnog modela i treniranih modela na istim podatcima koji su koristeni zatreningPod-zadatak 11 - Usporedna evaluacija zadanog modela, slucajnog modela i treniranih modela na istim podatcima koji su koristeni za trening\n");

% Izračuni log-izvjenosti kako bi usporedili uspjesnost modeliranja 
% opservacijskih nizova generiranih u pod-zadatku 7 sa svim raspoloživim 
% HMM modelima

% Log-izvjesnost za zadani model
ll_zadani_model=dhmm_logprob(data, prior0, transmat0, obsmat0);
% Ispis za zadani model
fprintf('Log-izvjenost za zadani model: %f\n', ll_zadani_model);

% Log-izvjesnost za "los" model
ll_los_model=dhmm_logprob(data, prior1, transmat1, obsmat1);
% Ispis za "los" model
fprintf('Log-izvjenost za "los" model: %f\n', ll_los_model);

% Log-izvjesnost za HMM1 model (proizlazi iz loseg modela)
ll_HMM1_model=dhmm_logprob(data, prior2, transmat2, obsmat2);
% Ispis za HMM1 model
fprintf('Log-izvjenost za HMM1 model: %f\n', ll_HMM1_model);

% Log-izvjesnost za HMM2 model (proizlazi iz zadanog modela)
ll_HMM2_model=dhmm_logprob(data, prior3, transmat3, obsmat3);
% Ispis za HMM2 model
fprintf('Log-izvjenost za HMM2 model: %f\n', ll_HMM2_model);

% Ispis parametara novog modela treniranog na zadanom modelu
fprintf('Matrica A - trenirani model (prijelazne vjerojatnosti):\n')
disp(transmat3)
fprintf('Matrica B - trenirani model (emisijske vjerojatnosti):\n')
disp(obsmat3)
fprintf('Matrica pi - trenirani model (početne vjerojatnosti):\n')
disp(prior3)