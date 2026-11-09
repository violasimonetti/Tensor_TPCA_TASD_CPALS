function metrics = loading_metrics(Uhat, Atrue, recoveryTol)
%LOADING_METRICS Compare CP loading directions using one-to-one matching.
%
% Inputs contain one factor matrix per mode. Columns are normalized before
% comparison. Signs may differ, but matching is common to all modes.
%
% error        : exact minimax error; NaN when the CP ranks differ
% matchedError : minimax error for min(Rfit, Rtrue) matched components
% recall       : number of accurately matched components / Rtrue
% precision    : number of accurately matched components / Rfit
% f1           : harmonic mean of component recall and precision
% success      : all components accurately recovered at the correct rank
% valid        : estimated loading columns are real, finite and nonzero
%
% Invalid estimates receive error sqrt(2), zero recall/precision and failure.
% For unequal ranks, read matchedError together with recall and precision.

    if nargin < 3
        recoveryTol = 0.05;
    end

    validateattributes(recoveryTol, {'numeric'}, ...
        {'real', 'scalar', 'finite', 'nonnegative', '<', sqrt(2)});

    d = numel(Atrue);
    if ~iscell(Uhat) || ~iscell(Atrue) || d == 0 || numel(Uhat) ~= d
        error('loading_metrics:ModeMismatch', ...
            'Inputs must be nonempty cell arrays with the same modes.');
    end

    Rtrue = size(Atrue{1}, 2);
    Rfit = size(Uhat{1}, 2);
    if Rtrue == 0 || Rfit == 0
        error('loading_metrics:EmptyRank', 'Both CP ranks must be positive.');
    end

    metrics = struct;
    metrics.error = NaN;
    if Rfit == Rtrue
        metrics.error = sqrt(2);
    end
    metrics.matchedError = sqrt(2);
    metrics.meanMatchedError = sqrt(2);
    metrics.modeError = sqrt(2) * ones(d, 1);
    metrics.componentError = nan(1, Rtrue);
    metrics.recall = 0;
    metrics.precision = 0;
    metrics.f1 = 0;
    metrics.nRecovered = 0;
    metrics.success = false;
    metrics.valid = false;
    metrics.pairs = zeros(0, 2);
    metrics.recoveredPairs = zeros(0, 2);

    estimatedValid = true;
    for k = 1:d
        A = Atrue{k};
        U = Uhat{k};
        if ~isnumeric(A) || ~isnumeric(U) || ...
                size(A, 2) ~= Rtrue || size(U, 2) ~= Rfit || ...
                size(A, 1) ~= size(U, 1)
            error('loading_metrics:SizeMismatch', ...
                'Factor matrices must have compatible rows and ranks.');
        end

        aNorm = vecnorm(A, 2, 1);
        uNorm = vecnorm(U, 2, 1);
        if ~isreal(A) || any(~isfinite(A(:))) || ...
                any(~isfinite(aNorm)) || any(aNorm == 0)
            error('loading_metrics:InvalidTruth', ...
                'True loading columns must be real, finite and nonzero.');
        end
        if ~isreal(U) || any(~isfinite(U(:))) || ...
                any(~isfinite(uNorm)) || any(uNorm == 0)
            estimatedValid = false;
        else
            Uhat{k} = U ./ uNorm;
        end
        Atrue{k} = A ./ aNorm;
    end

    if ~estimatedValid
        return;
    end

    % A pair is close only if its loading vectors are close in every mode.
    modeCost = zeros(Rfit, Rtrue, d);
    for k = 1:d
        for i = 1:Rfit
            for r = 1:Rtrue
                u = Uhat{k}(:, i);
                a = Atrue{k}(:, r);
                modeCost(i, r, k) = min(norm(u - a), norm(u + a));
            end
        end
    end
    costMat = max(modeCost, [], 3);

    % The first feasible threshold gives the exact bottleneck assignment.
    thresholds = unique(costMat(:));
    nPairs = min(Rfit, Rtrue);
    for t = 1:numel(thresholds)
        pairs = maximum_matching(costMat <= thresholds(t));
        if size(pairs, 1) == nPairs
            metrics.pairs = pairs;
            metrics.matchedError = thresholds(t);
            break;
        end
    end

    pairIndex = sub2ind([Rfit Rtrue], pairs(:, 1), pairs(:, 2));
    metrics.componentError(pairs(:, 2)) = costMat(pairIndex);
    metrics.meanMatchedError = mean(costMat(pairIndex));
    for k = 1:d
        distances = modeCost(:, :, k);
        metrics.modeError(k) = max(distances(pairIndex));
    end

    % Threshold matching maximizes the count recovered. This assignment can
    % differ from the bottleneck assignment measuring continuous error.
    metrics.recoveredPairs = maximum_matching(costMat <= recoveryTol);
    metrics.nRecovered = size(metrics.recoveredPairs, 1);
    metrics.recall = metrics.nRecovered / Rtrue;
    metrics.precision = metrics.nRecovered / Rfit;
    metrics.f1 = 2 * metrics.nRecovered / (Rtrue + Rfit);
    metrics.valid = true;
    if Rfit == Rtrue
        metrics.error = metrics.matchedError;
        metrics.success = metrics.error <= recoveryTol;
    end

end

function pairs = maximum_matching(allowed)
% Maximum-cardinality bipartite matching without extra toolboxes.

    [Rfit, Rtrue] = size(allowed);
    estimatedForTrue = zeros(1, Rtrue);
    for i = 1:Rfit
        visited = false(1, Rtrue);
        [~, estimatedForTrue, ~] = augment_matching( ...
            i, allowed, estimatedForTrue, visited);
    end

    trueIndex = find(estimatedForTrue > 0);
    pairs = [estimatedForTrue(trueIndex)', trueIndex'];

end

function [found, estimatedForTrue, visited] = augment_matching( ...
        i, allowed, estimatedForTrue, visited)
% Reassign an existing pair when this makes room for another component.

    found = false;
    for r = find(allowed(i, :))
        if visited(r)
            continue;
        end
        visited(r) = true;
        if estimatedForTrue(r) == 0
            canAssign = true;
        else
            [canAssign, estimatedForTrue, visited] = augment_matching( ...
                estimatedForTrue(r), allowed, estimatedForTrue, visited);
        end
        if canAssign
            estimatedForTrue(r) = i;
            found = true;
            return;
        end
    end

end
