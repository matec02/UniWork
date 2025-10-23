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