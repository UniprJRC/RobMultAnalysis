load cows2026
X2=cows2026{:,:};
disp('Correlation matrix with listwise deletion')
R1=corr(X2,'Rows','complete');
disp(R1)
disp('Correlation matrix with pairwise deletion')
R2=corr(X2,'Rows','pairwise');
disp(R2)
prin=false;
[n,p]=size(X2);

[XcompleteCases,idxRemovedCows]=rmmissing(X2);

%% EM TEM application: input data for index plot of MD 
% alpha = 0, (no trimming) creates Figure 4 and alpha = 0.5 creates Figure 5

alpha=0.5;

if alpha==0
    outEM = mdEM(X2);
else
    outEM = mdTEM(X2,'method','pri','alpha',alpha);
end
muHat = outEM.loc;
SigHat = outEM.cov;
[d2_part, poss] = mdPartialMD(X2, muHat, SigHat);
d2_pri =mdPartialMD2full(d2_part, p, poss, 'method','pri');

%% Index plot of robust Mahalanobis distances (Figure 5) 
malindexplot(d2_pri,p,'conflev',0.95,'numlab',{8})
xlim([0 n+1])
title('')
ylabel(['MD alpha=' num2str(alpha)])
xlabel('Unit number')

if prin == true
    exportgraphics(gcf,'figs/MDindex050.pdf');
    % exportgraphics(gcf,'figs/MDindex000.pdf')
end

%% Scatter plot matrix with outliers  (Figure 6)
figure
bool=d2_pri>chi2inv(0.95,p);
typespm="upper";
spmplot(X2,"group",bool,"typespm",typespm);
disp('Number of units decleared as outliers')
disp(sum(bool))

if prin == true
    exportgraphics(gcf,'figs/spmplot050.pdf');
end

%% Compare original distances for complete rows before and after EM (Figure 8)
figure
d2OriCows=mahalFS(XcompleteCases,mean(XcompleteCases),cov(XcompleteCases));

Alpha=[0 0.5];
for j=1:2
    subplot(2,2,j)
    alpha=Alpha(j);

    if alpha==0
        outEM = mdEM(X2);
    else
        outEM = mdTEM(X2,'method','pri','alpha',alpha);
    end

    muHat = outEM.loc;
    SigHat = outEM.cov;
    [d2_part, poss] = mdPartialMD(X2, muHat, SigHat);

    d2_pri =mdPartialMD2full(d2_part, p, poss, 'method','pri');
    d2_priAUG=d2_pri;
    d2_priAUG(idxRemovedCows)=[];
    scatter(d2OriCows,d2_priAUG)
    refline(1)
    xlabel('Original distance')
    % ylabel('Distance from EM')
    ylabel(['\alpha=' num2str(alpha)])
    box on
end
if prin == true
    exportgraphics(gcf,'figs/testMCARcows.pdf');
end

%% Number of outliers as function of level of trimming (Table 1)
tol=1e-10;
close all
Alpha=[0.5 0.25 0.20 0.15 0.10 0.05 0.01 0]';
methods=["pri" "betaMap"];
numout=zeros(length(Alpha),2);
for j=1:length(methods)
for i=1:length(Alpha)

    if Alpha(i)==0
        outEM = mdEM(X2);
    else
        outEM = mdTEM(X2,'method',methods(j),'alpha',Alpha(i),'tol',tol);
    end

    muHat = outEM.loc;
    SigHat = outEM.cov;
    [d2_part, poss] = mdPartialMD(X2, muHat, SigHat);

    d2_pri =mdPartialMD2full(d2_part, p, poss, 'method','pri');

    bool=d2_pri>chi2inv(0.95,p);
    numout(i,j)=sum(bool);
end
end

AlphaNout=array2table(numout,"RowNames",string(Alpha),"VariableNames",methods);
disp('Number of units above Chi2 0.95 threshold for different alpha with 2 rescaling methods')
disp(AlphaNout)
% If plot is needed uncomment what is below
%{
    plot(Alpha,numout)
    xlabel('Trimming level')
    ylabel('Number of units declared outliers')
    set(gca,"XDir","reverse")
    legend(methods)
%}


%% Creation of Figure with 4 panels which compares  original distances for 
% complete rows before and after EM (Figure 7)

figure
rng(11)
p = 7;                % number of variables
n = 500;              % number of observations
rho = 0.5;            % target pairwise correlation (0<rho<1)
Sigma = (1-rho)*eye(p) + rho*ones(p);
R = chol(Sigma);      % upper-triangular such that Sigma = R'*R
missRate = 0.09;      % MCAR missing probability per entry

% Generate samples ~ N(0,Sigma)
Yfull = randn(n,p) * R;   % Strong positive correlation between the vars
missMask = rand(n,p) < missRate;
mu=mean(Yfull)';
S=cov(Yfull);
Y=Yfull;
Y(missMask) = NaN;
tol=1e-12;

[XcompleteCaseSIM,idxRemoved]=rmmissing(Y);
d2Ori=mahalFS(XcompleteCaseSIM,mean(XcompleteCaseSIM),cov(XcompleteCaseSIM));

figure
Alpha=[0 0.1 0.25 0.5];
for j=1:4
    subplot(2,2,j)
    alpha=Alpha(j);

    if alpha==0
        outEM = mdEM(Y);
    else
        outEM = mdTEM(Y,'method','pri','alpha',alpha);
    end

    muHat = outEM.loc;
    SigHat = outEM.cov;
    [d2_part, poss] = mdPartialMD(Y, muHat, SigHat);

    d2_pri =mdPartialMD2full(d2_part, p, poss, 'method','pri');

    d2_priAUG=d2_pri;
    d2_priAUG(idxRemoved)=[];
    scatter(d2Ori,d2_priAUG)
    refline(1)
    xlabel('Original distance')
    ylabel(['\alpha=' num2str(alpha)])
    box on
end

if prin == true
    exportgraphics(gcf,'figs/testMCAR.pdf')
end

%% Call function which implements the test
% To replicate exactly Figure 9, the parameter 'nsimul' of the function
% mdMCARtest() should be set to 10000.

out = mdMCARtest(X2, 'alpha', 0.5,'method', 'pri', ...
    'nsimul', 199, 'plots', true);
disp(out.pvalue)
disp(out.Tobs)
disp(out.ciBoot)

if prin == true
    exportgraphics(gcf,'figs/testMCARpval.pdf')
end