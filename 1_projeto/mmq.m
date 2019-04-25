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

[_, n_entries,] = size(X);

Pacertos = [];

for i = 1:1000
  I=randperm(n_entries);

  % Embaralha dados
  X=X(:,I);
  Y=Y(:,I);

  % 80/20
  eighty = floor(n_entries * (80/100));

  Xmodel = X(:, 1:eighty);
  Ymodel = X(:, 1:eighty);

  Xtest = X(:, (eighty + 1):end);
  Ytest = Y(:, (eighty + 1):end);

  % Construcao do modelo (Determinacao da matriz A)
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
  Pacertos(:, end + 1) = Pacerto;
endfor

Pacertos_maximum = max(Pacertos)
Pacertos_minimum = min(Pacertos)
Pacertos_mean = mean(Pacertos)










