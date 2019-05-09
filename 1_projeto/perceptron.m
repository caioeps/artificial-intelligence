clear; clc;

pkg load 'statistics';

%% Load CSV
dataset = dlmread('./lung-cancer.data', ',');

%% Labels
Y = dataset(:,1);
%% Attributes
X = dataset(:,2:end);

%% Remove corrupted data
X(:,[5,39]) = [];

X = X';
Y = Y';

[n_attributes n_entries] = size(X);

Ynormalized = [];

%% [1,3,2]
%% becomes
%% 1 0 0
%% 0 0 1
%% 0 1 0
for i = 1:n_entries
  a = [0; 0; 0];
  a(Y(i),:) = 1;
  Ynormalized(:,i) = a;
endfor

Y = Ynormalized;

%% Normaliza componentes para media zero e variancia unitaria
for i=1:n_attributes,
	m = mean(X(i,:));  % Media das linhas
  d = std(X(i,:));   % desvio-padrao das linhas
	X(i,:) = (X(i,:) - m) ./ d;
end

%% DEFINE ARQUITETURA DA REDE
%=========================
n_epochs        = 10;
n_rounds        = 100;
n_output_neuron = 3;    % No. de neuronios na camada de saida
ptrn            = 0.97;  % Porcentagem usada para treino
eta             = 0.1; % Passo de aprendizagem

%% Inicio do Treino
for roundNumber = 1:n_rounds,  % LOOP DE RODADAS TREINO/TESTE
  I = randperm(n_entries);
  X = X(:,I);
  Y = Y(:,I);   % Embaralha saidas desejadas tambem p/ manter correspondencia com vetor de entrada

  train_quantity = floor(ptrn * n_entries);

  %% Split train data
  train_X = X(:,1:train_quantity);
  train_Y = Y(:,1:train_quantity);
  [train_nAttributes train_nEntries] = size(train_X);

  %% Split test data
  test_X = X(:, train_quantity+1:end);
  test_Y = Y(:, train_quantity+1:end);
  [_ test_nEntries] = size(test_X);

  %% Initially random weight matrix
  W = 0.1 * rand(n_output_neuron, train_nAttributes+1);

  for epoch = 1:n_epochs,
    %% Shuffle train data.
    I = randperm(train_nEntries);
    train_X = train_X(:,I);
    train_Y = train_Y(:,I);
    train_quadraticError = 0;

    for attribute = 1:train_nEntries,
      %% Train network.
      train_Input = [-1; train_X(:,attribute)]; % x0 is -1;
      Yi = sign(W * train_Input); % Neurons network output activation

      %% Calculate error.
      train_err = train_Y(:,attribute) - Yi;
      train_quadraticError += eta * sum(train_err.^2);

      W += eta * train_err * train_Input';
    end

    train_quadraticErrorMean(epoch) = train_quadraticError / train_nEntries;
  end

  test_output = [];
  test_quadraticError = 0;
  for attribute = 1:test_nEntries,
    %% Test Generated Weight matrix.
    test_Input = [-1; test_X(:,attribute)];
    test_Yi = sign(W * test_Input);
    test_output = [test_output test_Yi];

    %% Calculate error.
    test_err = test_Y(:,attribute) - test_Yi;
    test_quadraticError += eta * sum(test_err.^2);
  end

  test_quadraticErrorMean = test_quadraticError / test_nEntries;

  test_error = test_Y - test_output;
  test_errorSum = sum(abs(test_error));
  test_nErrors = length(find(test_errorSum ~= 0));

  test_errorTax(roundNumber) = 100 * (test_nErrors / test_nEntries);
  test_successTax(roundNumber) = 100 - test_errorTax(roundNumber);
end

meanTax = mean(test_successTax)
standardDeviationTax = std(test_successTax)
medianTax = median(test_successTax)
minTax = min(test_successTax)
maxTax = max(test_successTax)

figure(1);
histfit(test_successTax);
figure(2);
boxplot(test_successTax);
figure(3);
plot(test_quadraticErrorMean);
