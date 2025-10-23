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