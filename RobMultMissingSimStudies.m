%% sim_compare_MD_corrections_EM.m
% Monte Carlo comparison of partial-MD corrections under missingness,
% using em_NA to estimate mu and Sigma each replicate.
%
% Methods compared:
%   1) impMD   : MD on EM-imputed data
%   2) pri     : principled EM rescaling = d2_partial + (p - p1)
%   3) expScale: d2 * (p/p1)
%   4) zMap    : p + sqrt(2p) * ((d2 - p1)/sqrt(2p1))
%   5) detMap  : determinant-based rescaling
%   6) chiMap  : MD_partial_adjusted (chi-square mapping)
%   7) betaMap : MD_partial_adjusted_beta
%
% Output:
%   - Summary table with mean metrics + Monte Carlo SE
%   - Figures: boxplots (RMSE/MAE), win-rate bar, MAE vs poss
%

clear; close all; clc;

%% ---------------- USER SETTINGS ----------------
rng(1);
prin=false;          % control whether to write the output to file
nsimul   = 5000;      % number of Monte Carlo replicates
n        = 50;       % sample size
p        = 15;       % dimension
missRate = 0.5;      % MCAR missing probability per entry
forceAtLeast1Obs = true;

alphaCut = 0.975;    % optional exceedance check versus chi2_p cutoff
cut = chi2inv(alphaCut, p);
useTrimming=false;
alpha=0.5;

highCorrelation=false; % control correlation level among variables

if highCorrelation==false
    % True model (choose something correlated)
    A = randn(p);
    SigmaTrue = A'*A;
    D = diag(1 ./ sqrt(diag(SigmaTrue)));
    SigmaTrue = D * SigmaTrue * D;      % "correlation-like"
    muTrue = linspace(-1,1,p);
else
    % target pairwise correlation (0<rho<1)
    rho = 0.9;
    % Covariance matrix (unit variances)
    SigmaTrue = (1-rho)*eye(p) + rho*ones(p);
    muTrue=linspace(100,200,p);
end

methods = {'impMD','pri','expScale','zMap','detMap','chiMap','betaMap'};
K = numel(methods);

% Store replicate-level metrics
RMSE = nan(nsimul,K);
MAE  = nan(nsimul,K);
BIAS = nan(nsimul,K);
EXCDIFF = nan(nsimul,K);   % |P(est>cut)-P(true>cut)| within replicate
WIN = zeros(nsimul,1);     % index of winning method by RMSE

% For MAE vs poss aggregated across replicates
% (accumulate sums and counts)
maxPoss = p;
absErrSum_byPoss = zeros(maxPoss,K);
absErrN_byPoss   = zeros(maxPoss,K);

%% ---------------- MONTE CARLO LOOP ----------------
for s = 1:nsimul
    % --- generate complete data
    Xfull = mvnrnd(muTrue', SigmaTrue, n);             % n x p
    d2_true = mahalFS(Xfull, muTrue, SigmaTrue);      % true full MD^2

    % --- induce MCAR missingness
    Y = Xfull;
    missMask = rand(n,p) < missRate;
    Y(missMask) = NaN;

    if forceAtLeast1Obs == true
        for i=1:n
            if all(isnan(Y(i,:)))
                j = randi(p);
                Y(i,j) = Xfull(i,j);
            end
        end
    end

    if useTrimming==true
        % ============================================================
        % TRIMMED CASE
        % ============================================================

        outEM = mdTEM(Y,'condmeanimp',true,'method','impMD','alpha',alpha);
        muHat = outEM.loc;
        SigHat = outEM.cov;


        % --- 1) impMD
        Yimp = outEM.Yimp;
        d2_imp = mahalFS(Yimp, muHat', SigHat);

        % All the methods below require the rescaling of partial MD
        % --- partial MD using estimated mu/Sigma

        % --- 2) principled EM rescaling
        outEM = mdTEM(Y,'method','pri','alpha',alpha);
        muHat = outEM.loc;
        SigHat = outEM.cov;
        [d2_part, poss] = mdPartialMD(Y, muHat, SigHat);

        d2_pri =mdPartialMD2full(d2_part, p, poss, 'method','pri');

        % --- 3) expScale
        outEM = mdTEM(Y,'method','expScale','alpha',alpha);
        muHat = outEM.loc;
        SigHat = outEM.cov;
        [d2_part, poss] = mdPartialMD(Y, muHat, SigHat);
        d2_exp =mdPartialMD2full(d2_part, p, poss, 'method','expScale');

        % --- 4) zMap
        outEM = mdTEM(Y,'method','zMap','alpha',alpha);
        muHat = outEM.loc;
        SigHat = outEM.cov;
        [d2_part, poss] = mdPartialMD(Y, muHat, SigHat);
        d2_z = mdPartialMD2full(d2_part, p, poss, 'method','zMap');

        % --- 5) detMap
        outEM = mdTEM(Y,'method','detMap','alpha',alpha);
        muHat = outEM.loc;
        SigHat = outEM.cov;
        [d2_part, poss] = mdPartialMD(Y, muHat, SigHat);
        d2_det = mdPartialMD2full(d2_part, p, poss, 'method','detMap','Y',Y,'Sigma',SigHat);

        % --- 6) chiMap
        outEM = mdTEM(Y,'method','chiMap','alpha',alpha);
        muHat = outEM.loc;
        SigHat = outEM.cov;
        [d2_part, poss] = mdPartialMD(Y, muHat, SigHat);
        d2_chi  = mdPartialMD2full(d2_part, p, poss, 'method','chiMap');

        % --- 7) betaMap
        outEM = mdTEM(Y,'method','betaMap','alpha',alpha);
        muHat = outEM.loc;
        SigHat = outEM.cov;
        [d2_part, poss] = mdPartialMD(Y, muHat, SigHat);
        d2_beta = mdPartialMD2full(d2_part, p, poss, 'method','betaMap');

    else
        % ============================================================
        % NO TRIMMING
        % ============================================================
        outEM = mdEM(Y,'condmeanimp',true);

        muHat = outEM.loc;
        SigHat = outEM.cov;


        % --- 1) impMD
        Yimp = outEM.Yimp;
        d2_imp = mahalFS(Yimp, muHat', SigHat);

        % All the methods below require the rescaling of partial MD
        % --- partial MD using estimated mu/Sigma
        [d2_part, poss] = mdPartialMD(Y, muHat, SigHat);

        % --- 2) principled EM rescaling
        d2_pri =mdPartialMD2full(d2_part, p, poss, 'method','pri');

        % --- 3) expScale
        d2_exp =mdPartialMD2full(d2_part, p, poss, 'method','expScale');

        % --- 4) zMap
        d2_z = mdPartialMD2full(d2_part, p, poss, 'method','zMap');

        % --- 5) detMap
        d2_det = mdPartialMD2full(d2_part, p, poss, 'method','detMap','Y',Y,'Sigma',SigHat);

        % --- 6) chiMap
        d2_chi  = mdPartialMD2full(d2_part, p, poss, 'method','chiMap');

        % --- 7) betaMap
        d2_beta = mdPartialMD2full(d2_part, p, poss, 'method','betaMap');
    end

    D2 = [d2_imp, d2_pri, d2_exp, d2_z, d2_det, d2_chi, d2_beta];

    % --- replicate metrics (compare to true d2_true)
    for k=1:K
        d2k = D2(:,k);
        ok = ~isnan(d2k) & ~isnan(d2_true);

        err = d2k(ok) - d2_true(ok);
        RMSE(s,k) = sqrt(mean(err.^2));
        MAE(s,k)  = mean(abs(err));
        BIAS(s,k) = mean(err);

        fracTrue = mean(d2_true(ok) > cut);
        fracEst  = mean(d2k(ok)      > cut);
        EXCDIFF(s,k) = abs(fracEst - fracTrue);

        % accumulate MAE vs poss
        poss_ok = poss(ok);
        absErr = abs(err);
        for pp=1:maxPoss
            sel = poss_ok==pp;
            if any(sel)
                absErrSum_byPoss(pp,k) = absErrSum_byPoss(pp,k) + sum(absErr(sel));
                absErrN_byPoss(pp,k)   = absErrN_byPoss(pp,k)   + sum(sel);
            end
        end
    end

    % winner by RMSE (smallest)
    [~,WIN(s)] = min(RMSE(s,:));
end

%% ---------------- SUMMARY TABLE ----------------
meanRMSE = mean(RMSE,1,'omitnan');
meanMAE  = mean(MAE,1,'omitnan');
meanBIAS = mean(BIAS,1,'omitnan');
meanEXC  = mean(EXCDIFF,1,'omitnan');

% Monte Carlo standard errors of the mean
seRMSE = std(RMSE,0,1,'omitnan')/sqrt(nsimul);
seMAE  = std(MAE,0,1,'omitnan')/sqrt(nsimul);
seBIAS = std(BIAS,0,1,'omitnan')/sqrt(nsimul);
seEXC  = std(EXCDIFF,0,1,'omitnan')/sqrt(nsimul);

winCounts = accumarray(WIN,1,[K 1],@sum,0);
winRate = winCounts / nsimul;

T = table(string(methods(:)), ...
    meanRMSE(:), seRMSE(:), ...
    meanMAE(:),  seMAE(:), ...
    meanBIAS(:), seBIAS(:), ...
    meanEXC(:),  seEXC(:), ...
    winCounts(:), winRate(:), ...
    'VariableNames', {'Method', ...
    'RMSE_mean','RMSE_se', ...
    'MAE_mean','MAE_se', ...
    'Bias_mean','Bias_se', ...
    'AbsExceedDiff_mean','AbsExceedDiff_se', ...
    'WinCount_RMSE','WinRate_RMSE'});

disp('=== Monte Carlo summary (smaller is better) ===');
disp(T);

%% ---------------- PLOTS ----------------
% Boxplot RMSE
figure('Name','RMSE across replicates','Color','w');
boxplot(RMSE, 'Labels', methods);
grid on;
ylabel('RMSE of d^2');
if useTrimming==false
    title(sprintf('RMSE across %d replicates (n=%d, p=%d, missRate=%.2f)', nsimul, n, p, missRate));
else
    title(sprintf('RMSE across %d replicates (n=%d, p=%d, missRate=%.2f, alpha=%.2f)', nsimul, n, p, missRate,alpha));
end

% Boxplot MAE
figure('Name','MAE across replicates','Color','w');
boxplot(MAE, 'Labels', methods);
grid on;
ylabel('MAE of d^2');
title(sprintf('MAE across %d replicates (n=%d, p=%d, missRate=%.2f)', nsimul, n, p, missRate));

% Win-rate bar (RMSE)
figure('Name','Win rate by RMSE','Color','w');
bar(winRate);
set(gca,'XTickLabel',methods);
grid on;
ylabel('Win rate (smallest RMSE)');
title('Winning method frequency (by RMSE)');

% Mean |error| vs poss
mae_byPoss = absErrSum_byPoss ./ max(absErrN_byPoss,1); % avoid divide-by-zero
figure('Name','MAE vs poss','Color','w');
hold on;
uPoss = (1:p)';
for k=1:K
    plot(uPoss, mae_byPoss(:,k), '-o', 'LineWidth', 1.2, 'MarkerSize', 4);
end
grid on;
xlabel('poss (observed dims p_1)');
ylabel('Mean |error|');
legend(methods,'Location','northeast');
title('Mean absolute error vs number of observed variables (aggregated over replicates)');

%% ---------------- OPTIONAL: SAVE TABLE ----------------
if prin==true
if useTrimming ==true
    str=['save Tnsimul' num2str(nsimul)  'n' num2str(n) 'p' num2str(p) 'miss' num2str(100*missRate) 'alpha' num2str(alpha) '.mat'];
    eval(str)
else
    str=['save nsimul' num2str(nsimul)  'n' num2str(n) 'p' num2str(p) 'miss' num2str(100*missRate) ];
    eval(str)
end
end