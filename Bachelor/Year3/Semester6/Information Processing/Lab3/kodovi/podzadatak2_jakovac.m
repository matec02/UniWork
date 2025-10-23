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
