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

[n_attributes, n_entries,] = size(X);

normalized_Y = [];

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

Y = normalized_Y;

% Normaliza componentes para media zero e variancia unitaria
for i=1:n_attributes,
	m = mean(X(i,:));  % Media das linhas
  d = std(X(i,:));   % desvio-padrao das linhas
	X(i,:) = (X(i,:) - m) ./ d;
end

% DEFINE ARQUITETURA DA REDE
%=========================
n_epochs        = 100;
n_rounds        = 10;
n_output_neuron = 6;    % No. de neuronios na camada de saida
ptrn            = 0.8;  % Porcentagem usada para treino
eta             = 0.01; % Passo de aprendizagem

%% Inicio do Treino
for rnd = 1:n_rounds,  % LOOP DE RODADAS TREINO/TESTE
  I = randperm(n_entries);
  X = X(:,I);
  Y = Y(:,I);   % Embaralha saidas desejadas tambem p/ manter correspondencia com vetor de entrada

  training_qty = floor(ptrn * n_entries);

  % Split train data
  Xtrain = X(:, 1:training_qty);
  Ytrain = Y(:, 1:training_qty);
  [n_train_attributes n_train_entries] = size(Xtrain);

  % Split test data
  Xtest = X(:, training_qty+1:end);
  Ytest = Y(:, training_qty+1:end);
  [_ n_test_entries] = size(Xtest);

  % Initially random weight matrix
  W = 0.1 * rand(n_output_neuron, n_train_attributes + 1);

  for epoch = 1:n_epochs,
    % Shuffle train data.
    I = randperm(n_train_entries);
    Xtrain = Xtrain(:, I);
    Ytrain = Ytrain(:, I);
    quadratic_error=0;

    for attribute = 1:n_train_entries,
      % Train network.
      Xentry = [-1; Xtrain(:, attribute)]; % x0 is -1;
      Yi = sign(W * Xentry); % Neurons network output activation

      % Calculate error.
      err = Ytrain(:, attribute) - Yi;
      quadratic_error += 0.5 * sum(Ei.^2);

      W += eta * err * Xentry';
    end

    QuadraticErrorMean(epoch) = quadratic_error / n_train_entries;
  end
end
