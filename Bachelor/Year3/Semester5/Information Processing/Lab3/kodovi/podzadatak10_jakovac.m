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