% Routines for opening face images and convert them to column vectors
% by stacking the columns of the face matrix one beneath the other.
%
% Last modification: 23/04/2018
% Author: Guilherme Barreto

clear; clc; close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fase 1 -- Carrega imagens disponiveis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
part1 = 'subject0';
part2 = 'subject';
part3 = {'.centerlight' '.glasses' '.happy' '.leftlight' '.noglasses' '.normal' '.rightlight' '.sad' '.sleepy' '.surprised' '.wink'};
part4 = strvcat(part3);

Nind=15;   % Quantidade de individuos (classes)
Nexp=length(part3);  % Quantidade de expressoes

X=[];  % Matriz que acumula imagens vetorizadas
D=[];  % Matriz que acumula o rotulo (identificador) do individuo
NAME=[];
for i=1:Nind,  % Indice para os individuos
    individuo=i,
    for j=1:Nexp,   % Indice para expressoes
        if i<10,
            nome = strcat(part1,int2str(i),part4(j,:));    % Monta o nome do arquivo de imagem
        else
            nome = strcat(part2,int2str(i),part4(j,:));
        end
        Img=imread(nome);  % le imagem
        % imshow(Img); pause;
        Ar = imresize(Img,[50 50]);   % Redimensionamento da imagem
        An=imnoise(Ar,'salt & pepper',0.05);  % imagem com ruido
        A=im2double(An);  % converte (im2double) para double precision

        a=var(A,[],2);  % Calcula variancia de cada linha, trasnformando matriz em vetor.
        
        ROT=-ones(Nind,1); ROT(i)=1;  % Cria rotulo (label) da imagem
        X=[X a]; % Coloca cada imagem vetorizada como coluna da matriz X
        D=[D ROT]; % Coloca o rotulo de cada vetor como coluna da matriz d
    end 
end

%%% Gerando diferentes conjuntos de treino/teste
% Modo 1: Adiciona ruido gaussiano as imagens de treino
Xtrn=X;
Dtrn=D;

ruido_gauss=0.05*randn(size(X));
Xtst=X+ruido_gauss;
Dtst=D;

% Modo 2: Retira uma das imagens de cada individuo
Xtrn=[X(:,1:10) X(:,12:21) X(:,23:32) X(:,34:43) X(:,45:54)];
Xtrn=[Xtrn X(:,56:65) X(:,67:76) X(:,78:87) X(:,89:98) X(:,100:109)];
Xtrn=[Xtrn X(:,111:120) X(:,122:131) X(:,133:142) X(:,144:153) X(:,155:164)];

Dtrn=[D(:,1:10) D(:,12:21) D(:,23:32) D(:,34:43) D(:,45:54)];
Dtrn=[Dtrn D(:,56:65) D(:,67:76) D(:,78:87) D(:,89:98) D(:,100:109)];
Dtrn=[Dtrn D(:,111:120) D(:,122:131) D(:,133:142) D(:,144:153) D(:,155:164)];

Xtst=[X(:,11) X(:,22) X(:,33) X(:,44) X(:,55) X(:,66) X(:,77) X(:,88)];
Xtst=[Xtst X(:,99) X(:,110) X(:,121) X(:,132) X(:,143) X(:,154) X(:,165)];

Dtst=[D(:,11) D(:,22) D(:,33) D(:,44) D(:,55) D(:,66) D(:,77) D(:,88)];
Dtst=[Dtst D(:,99) D(:,110) D(:,121) D(:,132) D(:,143) D(:,154) D(:,165)];