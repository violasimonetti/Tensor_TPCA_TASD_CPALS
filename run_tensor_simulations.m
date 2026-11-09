function results = run_tensor_simulations(config)
%RUN_TENSOR_SIMULATIONS Compare loading accuracy, computation and reliability.
%
% Tensor_simulations.m contains the full design. A smaller config can be used
% for validation without changing that script or its research results.
% Starts share a dataset and are not independent Monte Carlo observations.

    codeDir = fileparts(mfilename('fullpath'));
    addpath(fullfile(codeDir, 'tensor_toolbox-v3.8'));
    validate_config(config);
    conditions = make_conditions(config);
    methodNames = ["TASD", "TPCA"];
    C = numel(conditions);
    nMethods = numel(methodNames);
    runs = repmat(empty_run(true), config.M, C, nMethods, config.nStarts);
    withinDataDistance = nan(config.M, C, nMethods);
    trueLoadings = cell(config.M, C);

    if ~isempty(config.outputDir) && ~isfolder(config.outputDir)
        mkdir(config.outputDir);
    end

    experimentTimer = tic;
    for c = 1:C
        cond = conditions(c);
        for j = 1:config.M
            dataSeed = config.rngSeed + j;
            [X, Y, Atrue] = cp_tensor( ...
                config.Rtrue, numel(config.p), config.p, cond.xi, ...
                cond.lambda, 0, dataSeed, cond.sigma);
            trueLoadings{j, c} = Atrue;
            Xfull = double(full(X));
            Yfull = double(Y);
            signalNorm = norm(Xfull(:));
            realizedSNR = 20 * log10(signalNorm / norm(Yfull(:) - Xfull(:)));

            for b = 1:config.nStarts
                initSeed = config.rngSeed + 1000000 + ...
                    ((c - 1) * config.M + j - 1) * config.nStarts + b;

                % Balance method order to reduce systematic timing effects.
                methodOrder = 1:nMethods;
                if mod(j + b, 2) == 0
                    methodOrder = fliplr(methodOrder);
                end
                for m = methodOrder
                    run = fit_one_method(Y, Xfull, Atrue, ...
                        methodNames(m), cond.Rfit, initSeed, config);
                    run.dataSeed = dataSeed;
                    run.initSeed = initSeed;
                    run.signalNorm = signalNorm;
                    run.realizedSNR = realizedSNR;
                    runs(j, c, m, b) = run;
                end
            end

            for m = 1:nMethods
                sameDataRuns = reshape(runs(j, c, m, :), [], 1);
                withinDataDistance(j, c, m) = repeated_start_distance(sameDataRuns);
            end
        end

        results = struct;
        results.config = config;
        results.conditions = conditions;
        results.methodNames = methodNames;
        results.runs = runs;
        results.withinDataDistance = withinDataDistance;
        results.trueLoadings = trueLoadings;
        results.completedConditions = c;
        results.summaryTable = summarize_runs(results);
        if ~isempty(config.outputDir)
            save(fullfile(config.outputDir, 'simulation_results.mat'), ...
                'results', '-v7.3');
            writetable(results.summaryTable, fullfile(config.outputDir, 'summary.csv'));
        end
        fprintf('Completed %d/%d conditions | %s | %s | elapsed = %.1f sec\n', ...
            c, C, cond.study, cond.label, toc(experimentTimer));
    end

end

function run = fit_one_method(Y, Xtrue, Atrue, method, Rfit, initSeed, config)

    correctRank = Rfit == config.Rtrue;
    run = empty_run(correctRank);
    rng(initSeed);
    lastwarn('');
    initTimer = tic;
    try
        if method == "TASD"
            initResult = tasd_init_matlab(Y, Rfit);
        else
            initResult = tpca_init_matlab(Y, Rfit);
        end
        run.initSeconds = toc(initTimer);
        [~, run.initWarningID] = lastwarn;
    catch exception
        run.initSeconds = toc(initTimer);
        run.totalSeconds = run.initSeconds;
        run.status = "initialization failed";
        run.failureMessage = string(exception.identifier) + ": " + exception.message;
        return;
    end

    initial = loading_metrics(initResult.U, Atrue, config.recoveryTol);
    run.initialU = initResult.U;
    run.initialError = initial.error;
    run.initialMatchedError = initial.matchedError;
    run.initialRecall = initial.recall;
    run.initialPrecision = initial.precision;
    run.initialF1 = initial.f1;
    run.initialValid = initial.valid;
    if ~initial.valid
        run.totalSeconds = run.initSeconds;
        run.status = "invalid initialization";
        run.failureMessage = "Estimated columns are not real, finite and nonzero.";
        return;
    end

    lastwarn('');
    alsTimer = tic;
    try
        [P, ~, out] = cp_als(Y, Rfit, ...
            'init', initResult.U, ...
            'maxiters', config.maxit, ...
            'tol', config.alsTol, ...
            'printitn', 0);
        run.alsSeconds = toc(alsTimer);
        [~, run.alsWarningID] = lastwarn;
    catch exception
        run.alsSeconds = toc(alsTimer);
        run.totalSeconds = run.initSeconds + run.alsSeconds;
        run.status = "ALS failed";
        run.failureMessage = string(exception.identifier) + ": " + exception.message;
        return;
    end

    % Evaluation is excluded from initialization and ALS timings.
    run.totalSeconds = run.initSeconds + run.alsSeconds;
    run.alsIters = out.iters;
    run.hitIterationLimit = out.iters >= config.maxit;
    final = loading_metrics(P.U, Atrue, config.recoveryTol);
    run.finalU = P.U;
    if ~final.valid || any(~isfinite(P.lambda))
        run.status = "invalid final estimate";
        run.failureMessage = "The fitted CP model has invalid loadings or weights.";
        return;
    end

    Xhat = double(full(P));
    if any(~isfinite(Xhat(:)))
        run.status = "invalid reconstruction";
        run.failureMessage = "The fitted tensor has nonfinite entries.";
        return;
    end
    run.finalError = final.error;
    run.finalMatchedError = final.matchedError;
    run.finalMeanMatchedError = final.meanMatchedError;
    run.finalModeError = final.modeError;
    run.finalComponentError = final.componentError;
    run.finalRecall = final.recall;
    run.finalPrecision = final.precision;
    run.finalF1 = final.f1;
    run.recoverySuccess = double(final.success);
    if ~correctRank
        run.recoverySuccess = NaN;
    end
    run.reconError = norm(Xhat(:) - Xtrue(:)) / norm(Xtrue(:));
    run.failed = false;
    run.status = "completed";

end

function run = empty_run(correctRank)
% Failed attempts remain in error means and recovery-rate denominators.

    run = struct;
    run.initialError = sqrt(2);
    run.finalError = sqrt(2);
    run.recoverySuccess = 0;
    if ~correctRank
        run.initialError = NaN;
        run.finalError = NaN;
        run.recoverySuccess = NaN;
    end
    run.initialMatchedError = sqrt(2);
    run.finalMatchedError = sqrt(2);
    run.finalMeanMatchedError = sqrt(2);
    run.finalModeError = [];
    run.finalComponentError = [];
    run.initialRecall = 0;
    run.initialPrecision = 0;
    run.initialF1 = 0;
    run.finalRecall = 0;
    run.finalPrecision = 0;
    run.finalF1 = 0;
    run.initialValid = false;
    run.failed = true;
    run.initSeconds = 0;
    run.alsSeconds = 0;
    run.totalSeconds = 0;
    run.alsIters = NaN;
    run.hitIterationLimit = false;
    run.reconError = Inf;
    run.initialU = {};
    run.finalU = {};
    run.dataSeed = NaN;
    run.initSeed = NaN;
    run.signalNorm = NaN;
    run.realizedSNR = NaN;
    run.status = "not run";
    run.failureMessage = "";
    run.initWarningID = '';
    run.alsWarningID = '';

end

function distance = repeated_start_distance(runs)
% Directly compare loading estimates from the same observed tensor.

    B = numel(runs);
    if B < 2
        distance = NaN;
        return;
    end
    distances = zeros(B * (B - 1) / 2, 1);
    pair = 0;
    for b = 1:B - 1
        for other = b + 1:B
            pair = pair + 1;
            if runs(b).failed || runs(other).failed
                distances(pair) = sqrt(2);
            else
                distances(pair) = factor_eps(runs(b).finalU, runs(other).finalU);
            end
        end
    end
    distance = mean(distances);

end

function summaryTable = summarize_runs(results)

    config = results.config;
    rows = struct([]);
    row = 0;
    for c = 1:results.completedConditions
        cond = results.conditions(c);
        for m = 1:numel(results.methodNames)
            row = row + 1;
            group = reshape(results.runs(:, c, m, :), config.M, config.nStarts);
            dataErrors = mean(reshape([group.finalError], ...
                config.M, config.nStarts), 2);
            dataSuccess = mean(reshape([group.recoverySuccess], ...
                config.M, config.nStarts), 2);
            iterValues = [group.alsIters];

            rows(row).study = cond.study;
            rows(row).label = cond.label;
            rows(row).method = results.methodNames(m);
            rows(row).sigma_used = cond.sigma;
            rows(row).xi_used = cond.xi;
            rows(row).Rfit_used = cond.Rfit;
            rows(row).lambda_min = min(cond.lambda);
            rows(row).lambda_max = max(cond.lambda);
            rows(row).meanInitialError = mean([group.initialError]);
            rows(row).meanFinalError = mean(dataErrors);
            rows(row).sdDataError = std(dataErrors);
            rows(row).q90FinalError = sample_quantile([group.finalError], 0.90);
            rows(row).meanInitialMatchedError = mean([group.initialMatchedError]);
            rows(row).meanFinalMatchedError = mean([group.finalMatchedError]);
            rows(row).meanInitialRecall = mean([group.initialRecall]);
            rows(row).meanFinalRecall = mean([group.finalRecall]);
            rows(row).meanInitialPrecision = mean([group.initialPrecision]);
            rows(row).meanFinalPrecision = mean([group.finalPrecision]);
            rows(row).meanInitialF1 = mean([group.initialF1]);
            rows(row).meanFinalF1 = mean([group.finalF1]);
            rows(row).recoveryProbability = mean(dataSuccess);
            rows(row).recoverySE = std(dataSuccess) / sqrt(config.M);
            rows(row).meanInitSeconds = mean([group.initSeconds]);
            rows(row).meanALSSeconds = mean([group.alsSeconds]);
            rows(row).meanTotalSeconds = mean([group.totalSeconds]);
            rows(row).meanALSItersCompleted = mean(iterValues(isfinite(iterValues)));
            rows(row).iterationLimitRate = mean([group.hitIterationLimit]);
            rows(row).failureRate = mean([group.failed]);
            rows(row).warningRate = mean( ...
                strlength(string({group.initWarningID})) > 0 | ...
                strlength(string({group.alsWarningID})) > 0);
            rows(row).meanWithinDataDistance = mean(results.withinDataDistance(:, c, m));
            rows(row).meanReconError = mean([group.reconError]);
            rows(row).meanSignalNorm = mean([group.signalNorm]);
            rows(row).meanRealizedSNR = mean([group.realizedSNR]);
        end
    end
    summaryTable = struct2table(rows);

end

function value = sample_quantile(values, probability)
% Interpolate order statistics without requiring the Statistics Toolbox.

    values = sort(values(:));
    if any(isnan(values))
        value = NaN;
        return;
    end
    position = 1 + (numel(values) - 1) * probability;
    lower = floor(position);
    upper = ceil(position);
    value = values(lower) + (position - lower) * (values(upper) - values(lower));

end

function conditions = make_conditions(config)

    conditions = struct('study', {}, 'label', {}, 'sigma', {}, ...
        'xi', {}, 'lambda', {}, 'Rfit', {});
    for sigma = config.sigmaGrid(:)'
        conditions(end + 1) = make_condition("noise_sigma", ...
            string(sprintf('sigma = %.0e', sigma)), sigma, ...
            config.baseXi, config.lambdaBase, config.Rtrue);
    end
    for q = 1:size(config.strengthLambda, 1)
        conditions(end + 1) = make_condition("factor_strength", ...
            config.strengthLabels(q), config.baseSigma, config.baseXi, ...
            config.strengthLambda(q, :), config.Rtrue);
    end
    for xi = config.xiGrid(:)'
        conditions(end + 1) = make_condition("coherence", ...
            string(sprintf('xi = %.2f', xi)), config.baseSigma, ...
            xi, config.lambdaBase, config.Rtrue);
    end
    for Rfit = config.RfitGrid(:)'
        conditions(end + 1) = make_condition("rank_misspecification", ...
            string(sprintf('Rfit = %d', Rfit)), config.baseSigma, ...
            config.baseXi, config.lambdaBase, Rfit);
    end

end

function cond = make_condition(study, label, sigma, xi, lambda, Rfit)

    cond.study = study;
    cond.label = label;
    cond.sigma = sigma;
    cond.xi = xi;
    cond.lambda = lambda;
    cond.Rfit = Rfit;

end

function validate_config(config)

    validateattributes(config.M, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(config.nStarts, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(config.maxit, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(config.alsTol, {'numeric'}, {'scalar', 'real', 'positive', 'finite'});
    validateattributes(config.recoveryTol, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'nonnegative', '<', sqrt(2)});
    validateattributes(config.Rtrue, {'numeric'}, {'scalar', 'integer', '>=', 2});
    validateattributes(config.p, {'numeric'}, {'vector', 'integer', 'positive'});
    validateattributes(config.rngSeed, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative', 'finite'});
    validateattributes(config.RfitGrid, {'numeric'}, {'vector', 'integer', 'positive'});
    validateattributes(config.lambdaBase, {'numeric'}, {'vector', 'positive', 'finite'});
    validateattributes(config.strengthLambda, {'numeric'}, {'2d', 'positive', 'finite'});
    validateattributes([config.baseSigma, config.sigmaGrid(:)'], ...
        {'numeric'}, {'real', 'positive', 'finite'});
    validateattributes([config.baseXi, config.xiGrid(:)'], ...
        {'numeric'}, {'real', '>=', 0, '<', 1, 'finite'});
    if numel(config.p) < 3 || min(config.p) < max([config.Rtrue, config.RfitGrid(:)'])
        error('run_tensor_simulations:Dimensions', ...
            'At least three modes are required, with sizes at least every CP rank.');
    end
    if numel(config.lambdaBase) ~= config.Rtrue || ...
            size(config.strengthLambda, 2) ~= config.Rtrue || ...
            numel(config.strengthLabels) ~= size(config.strengthLambda, 1)
        error('run_tensor_simulations:StrengthDesign', ...
            'Strength vectors and labels must agree with the true rank.');
    end

end
