clear; clc;

pkg load 'statistics';

% Load CSV
dataset = dlmread('./lung-cancer.data', ',');

% Labels
Y = dataset(:, 1);
% Attributes
X = dataset(:, 2:end);

% Remove corrupted data
X(:, [5,39]) = [];

X = X';
Y = Y';

[_, n_entries] = size(X);

normalized_Y = []

% [1,3,2]
% becomes
% 1 0 0
% 0 0 1
% 0 1 0
for i = 1:n_entries
  a = [0; 0; 0];
  a(Y(i), :) = 1;
  normalized_Y(:, i) = a;
endfor

Y = normalized_Y

Pacertos = [];

for i = 1:100
  I=randperm(n_entries);

  % Embaralha dados
  X=X(:,I);
  Y=Y(:,I);

  % 80/20
  % Leave one for testing
  eighty = floor(n_entries * (97/100));

  Xmodel = X(:, 1:eighty);
  Ymodel = X(:, 1:eighty);

  Xtest = X(:, (eighty + 1):end);
  Ytest = Y(:, (eighty + 1):end);

  % Construcao do modelo (Determinacao da matriz A)
  Ymodel
  A=Ymodel * Xmodel' * pinv(Xmodel * Xmodel');

  % Teste do modelo
  Ypred=A * Xtest;  % Diagnosticos preditos

  % Encontra os elementos de maior valor em cada coluna de Ypred
  [_ Imax_pred] = max(Ypred);

  % Encontra os elementos de maior valor em cada coluna de Ypred
  [_ Imax_test] = max(Ytest);

  % Calcula porcentagem de acerto
  Perro = 100 * length(find(Imax_pred-Imax_test ~= 0)) / length(Imax_pred);
  Pacerto= 100 - Perro;
  Pacertos(i) = Pacerto;
endfor

Pacertos_minimum = min(Pacertos)
Pacertos_mean = mean(Pacertos)
Pacertos_std = std(Pacertos)

figure(1);
histfit(Pacertos);
figure(2);
boxplot(Pacertos);
figure(3);
plot(Pacertos);











