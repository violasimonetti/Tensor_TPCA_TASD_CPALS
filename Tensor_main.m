%% MAIN
% This code checks convergence of CP-ALS decomposition via simulations
clc;
clear;

codeDir = fileparts(mfilename('fullpath'));
addpath(codeDir);
addpath(fullfile(codeDir, 'tensor_toolbox-v3.8'));

%%
% Paramaters
rngSeed = 42;
R = 5; % CP-rank
d = 4;
p = [22 22 84 15]; % d x 1
lambda = [5 4 3 2 1]; % R x 1

M = 50; % Number of simulations
% maxit = 50; % Maximum number of iterations
SNRdB = linspace(10,40, 10);
sigma = [10^(-9), 10^(-7), 10^(-5), 10^(-4), 10^(-2), 1];
xi = linspace(0, 0.9, 10);
N = size(xi,2);
S = size(sigma, 2);

% Initialize vectors
iters = zeros(N, S);
iters_MC = zeros(M,1);

% Generate random vectors for initialization
initSeed = 43;
rng(initSeed);

%%
% ALS stopping rule
alsTol = 1e-6;
maxit = 500;   % safeguard; ALS may stop much earlier

% Store final eps values
eps_final_TASD = nan(M, N, S);
eps_final_TPCA = nan(M, N, S);

tic;

total_jobs = S * N;
job_count = 0;

for s = 1:S

    for i = 1:N

        eps_TASD_local = nan(M,1);
        eps_TPCA_local = nan(M,1);

        parfor j = 1:M

            seed = rngSeed + j;

            % Generate tensor
            [~, Y, A] = cp_tensor( ...
                R, d, p, xi(i), lambda, SNRdB(s), seed, sigma(s));

            %% TASD initialization
            % Keep initialization randomness separate from tensor generation.
            rng(initSeed + j);
            init_TASD = tasd_init_matlab(Y, R);

            P_TASD = cp_als( ...
                Y, R, ...
                'init', init_TASD.U, ...
                'maxiters', maxit, ...
                'tol', alsTol, ...
                'printitn', 0);

            eps_TASD_local(j) = factor_eps(P_TASD.U, A);

            %% TPCA initialization
            init_TPCA = tpca_init_matlab(Y, R);

            P_TPCA = cp_als( ...
                Y, R, ...
                'init', init_TPCA.U, ...
                'maxiters', maxit, ...
                'tol', alsTol, ...
                'printitn', 0);

            eps_TPCA_local(j) = factor_eps(P_TPCA.U, A);

        end

        eps_final_TASD(:, i, s) = eps_TASD_local;
        eps_final_TPCA(:, i, s) = eps_TPCA_local;

        job_count = job_count + 1;

        fprintf(['Completed %d/%d grid points | s = %d/%d | i = %d/%d | ', ...
                 'elapsed = %.1f sec\n'], ...
                 job_count, total_jobs, s, S, i, N, toc);

    end
end

nbins = 20;

for s = 1:S

    figure('Name', sprintf('sigma = %g', sigma(s)));
    tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:N

        nexttile;

        % Log-scale errors
        eps_TASD = log10(max(eps_final_TASD(:, i, s), realmin));
        eps_TPCA = log10(max(eps_final_TPCA(:, i, s), realmin));

        % Use common bin edges for fair comparison
        all_eps = [eps_TASD; eps_TPCA];
        edges = linspace(min(all_eps), max(all_eps), nbins + 1);

        if edges(1) == edges(end)
            edges = edges(1) + linspace(-0.5, 0.5, nbins + 1);
        end

        histogram(eps_TASD, edges, ...
            'Normalization', 'probability', ...
            'FaceAlpha', 0.55, ...
            'DisplayName', 'TASD');

        hold on;

        histogram(eps_TPCA, edges, ...
            'Normalization', 'probability', ...
            'FaceAlpha', 0.55, ...
            'DisplayName', 'TPCA');

        title(sprintf('xi = %.2f', xi(i)));
        xlabel('log_{10}(final eps)');
        ylabel('Proportion');
        grid on;

        if i == 1
            legend('Location', 'best');
        end

    end

    sgtitle(sprintf('Final factor error histograms: sigma = %g', sigma(s)));

end

