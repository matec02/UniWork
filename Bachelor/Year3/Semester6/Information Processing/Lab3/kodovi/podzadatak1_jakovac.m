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