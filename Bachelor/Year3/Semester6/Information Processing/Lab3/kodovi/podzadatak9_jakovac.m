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