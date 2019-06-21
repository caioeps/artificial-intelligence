function [Y L V Q M]=myPCA(X,tol,mode)
%% 
%% Implementa tecnica PCA para reducao de dimensionalidade dos vetores que forma a matriz de dados X.
%% 
%% Obs: Cada coluna da matriz X corresponde a um padrao (vetor de entrada) cuja dimensao serah reduzida..
%% 
%% tol (0<tol<=1): variancia dos dados originais que deve ser mantida nos dados transformados Y.
%% L: autovalores da matriz de covariancia dos dados originais
%% V: autovetores da matriz de covariancia dos dados originais 
%% Q: quantidade de componentes principais que satisfazem tol.
%% M: matriz de transformacao
%% mode: extrai (mode=1) ou nao (mode=0) a media das imagens

n=size(X);

if mode==1,
	mi=mean(X,2);
	X=X-repmat(mi,1,n(2));
end

Cx=cov(X');  % matriz de covariancia dos dados em X
[V L]=eig(Cx);  
L=flipud(diag(L));  % Autovalores em ordem decrescente
V=fliplr(V);  % Autovetores ordenados do maior para menor autovalor

figure; bar(L);  % grafico da amplitude dos autovalores
title('Indice dos Autovalores em Ordem Decrescente de Amplitude')
xlabel('Autovalor'), ylabel('Magnitude');

SL=sum(L);  % Soma dos autovalores
aux=0;
for i=1:length(L),
    aux=aux+L(i);
    VE(i)=aux/SL;   % Variancia explicada
end
figure; plot(VE); grid

% escolha dos Q maiores autovalores (componentes principais)
Q=length(find(VE<=tol));

% Matriz de transformacao resultante
M=V(:,1:Q);

% Imagem transformada com blocos de tamanho K
Y=M'*X;
