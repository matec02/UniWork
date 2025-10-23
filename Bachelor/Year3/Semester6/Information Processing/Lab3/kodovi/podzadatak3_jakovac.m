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