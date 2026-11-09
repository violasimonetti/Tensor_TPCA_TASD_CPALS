function verify_loading_recovery()
%VERIFY_LOADING_RECOVERY Mathematical checks and a small end-to-end study.
% Validation outputs are kept separate from the full research results.

    codeDir = fileparts(mfilename('fullpath'));
    addpath(codeDir);
    addpath(fullfile(codeDir, 'tensor_toolbox-v3.8'));
    rng(123);

    %% Scaling, signs, permutation and common matching across modes
    A = {eye(4, 3), eye(5, 3), eye(6, 3)};
    U = cellfun(@(a) a(:, [3 1 2]) .* [-2 3 -4], A, 'UniformOutput', false);
    metrics = loading_metrics(U, A);
    assert(metrics.error < 1e-12 && metrics.success && metrics.valid);
    U{2} = U{2}(:, [2 3 1]);
    metrics = loading_metrics(U, A);
    assert(metrics.error > 1 && ~metrics.success);

    %% Exact minimax matching checked against exhaustive permutations
    for trial = 1:20
        Atest = {randn(5, 3), randn(6, 3), randn(7, 3)};
        Utest = {randn(5, 3), randn(6, 3), randn(7, 3)};
        metrics = loading_metrics(Utest, Atest, 0.8);
        [expectedError, expectedCount] = exhaustive_metrics(Utest, Atest, 0.8);
        assert(abs(metrics.error - expectedError) < 1e-12);
        assert(metrics.nRecovered == expectedCount);
    end

    %% Missing, extra and duplicate components
    under = cellfun(@(a) a(:, 1:2), A, 'UniformOutput', false);
    metrics = loading_metrics(under, A);
    assert(isnan(metrics.error) && metrics.matchedError == 0);
    assert(metrics.recall == 2/3 && metrics.precision == 1);
    over = cellfun(@(a) [a, a(:, 1)], A, 'UniformOutput', false);
    metrics = loading_metrics(over, A);
    assert(metrics.recall == 1 && metrics.precision == 3/4);
    duplicate = cellfun(@(a) a(:, [1 1 2]), A, 'UniformOutput', false);
    metrics = loading_metrics(duplicate, A);
    assert(metrics.nRecovered == 2 && ~metrics.success);

    %% Invalid outputs cannot silently count as successful recovery
    bad = A;
    bad{1}(:, 1) = 0;
    metrics = loading_metrics(bad, A);
    assert(~metrics.valid && ~metrics.success && metrics.error == sqrt(2));
    bad{1}(1, 1) = NaN;
    metrics = loading_metrics(bad, A);
    assert(~metrics.valid && metrics.recall == 0 && metrics.precision == 0);

    %% Small simulation covering every design and both fitted-rank errors
    config = struct;
    config.rngSeed = 42;
    config.Rtrue = 3;
    config.p = [6 7 8 5];
    config.lambdaBase = [5 3 1];
    config.M = 2;
    config.nStarts = 2;
    config.alsTol = 1e-6;
    config.maxit = 80;
    config.recoveryTol = 0.05;
    config.baseXi = 0.2;
    config.baseSigma = 1e-2;
    config.sigmaGrid = [1e-9 1e-2 1];
    config.xiGrid = [0 0.7];
    config.RfitGrid = [2 3 4];
    config.strengthLambda = [5 3 1; 1.25 0.75 0.25; 5 3 0.25];
    config.strengthLabels = ["strong", "all weak", "one weak"];
    config.outputDir = fullfile(codeDir, 'Validation_v3');
    results = run_tensor_simulations(config);
    assert(results.completedConditions == 11);
    assert(height(results.summaryTable) == 22);
    assert(all(results.summaryTable.meanTotalSeconds >= 0));
    assert(all(results.summaryTable.meanFinalRecall >= 0 & ...
        results.summaryTable.meanFinalRecall <= 1));
    assert(all(results.summaryTable.meanFinalPrecision >= 0 & ...
        results.summaryTable.meanFinalPrecision <= 1));

    for c = 1:numel(results.conditions)
        correctRank = results.conditions(c).Rfit == config.Rtrue;
        for j = 1:config.M
            for m = 1:2
                first = results.runs(j, c, m, 1);
                second = results.runs(j, c, m, 2);
                assert(first.dataSeed == second.dataSeed);
                assert(first.initSeed ~= second.initSeed);
                assert(abs(first.totalSeconds - first.initSeconds - first.alsSeconds) < 1e-9);
                if ~correctRank
                    assert(isnan(first.finalError) && isnan(first.recoverySuccess));
                end
                if m == 2 && ~first.failed && ~second.failed
                    assert(factor_eps(first.finalU, second.finalU) < 1e-12);
                end
            end
        end
    end

    % Verify reproducibility of estimates, excluding elapsed time.
    cond = results.conditions(1);
    [~, Y, Atrue] = cp_tensor(config.Rtrue, numel(config.p), config.p, ...
        cond.xi, cond.lambda, 0, config.rngSeed + 1, cond.sigma);
    saved = results.runs(1, 1, 1, 1);
    rng(saved.initSeed);
    init = tasd_init_matlab(Y, cond.Rfit);
    assert(factor_eps(init.U, saved.initialU) < 1e-12);
    assert(abs(factor_eps(init.U, Atrue) - saved.initialError) < 1e-12);

    plot_tensor_simulations(results, fullfile(config.outputDir, 'figures'));
    close all;
    fprintf('Loading recovery checks passed: invariance, exact matching, ranks, failures and 88 fits.\n');

end

function [bestError, bestCount] = exhaustive_metrics(U, A, tolerance)

    costs = zeros(3, 3);
    for k = 1:numel(A)
        U{k} = U{k} ./ vecnorm(U{k});
        A{k} = A{k} ./ vecnorm(A{k});
        for i = 1:3
            for r = 1:3
                distance = min(norm(U{k}(:, i) - A{k}(:, r)), ...
                    norm(U{k}(:, i) + A{k}(:, r)));
                costs(i, r) = max(costs(i, r), distance);
            end
        end
    end
    permutations = perms(1:3);
    bestError = Inf;
    bestCount = 0;
    for t = 1:size(permutations, 1)
        values = costs(sub2ind([3 3], 1:3, permutations(t, :)));
        bestError = min(bestError, max(values));
        bestCount = max(bestCount, sum(values <= tolerance));
    end

end
