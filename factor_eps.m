function eps_t = factor_eps(Uhat, Atrue)
%FACTOR_EPS Worst loading error after normalization, signs and permutation.
%
% Both inputs contain one factor matrix per mode and have the same CP rank.
% One common permutation minimizes the largest loading error across modes.
% Invalid estimated columns receive the worst unit-vector error, sqrt(2).

    if size(Uhat{1}, 2) ~= size(Atrue{1}, 2)
        error('factor_eps:RankMismatch', ...
            'Use loading_metrics for comparisons with unequal CP ranks.');
    end

    metrics = loading_metrics(Uhat, Atrue);
    eps_t = metrics.error;

end
