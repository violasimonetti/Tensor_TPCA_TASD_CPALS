function [X, Y, A] = cp_tensor(R, d, p, xi, lambda, SNRdB, rngSeed, sigma)
% This function computes a CP-rank tensor plus noise.
%
% Inputs:
% - d: (scalar) order of tensor
% - p: (dx1) vector of dimensions for mode-k matrix with k = 1,...,d
% - xi: (scalar) coherence parameter
% - lambda: (Rx1) scaling factors
% - SNRdB: (scalar) signal-to-noise ratio in db
% - rngSeed: (scalar) seed for reprudicibility
% - sigma: (scalar) noise level (put 0 if you prefer to use SNRdB)
%
% Ouputs:
% - X: (p(1) x ... x p(d)) CP-rank tensor
% - A: (p(k) x R) mode-k matrices
% - Y: (p(1) x ... x p(d)) CP-rank tensor plus noise

    rng(rngSeed);

    % Building factor matrices
    A = cell(1,d);

    for k = 1:d
        A{k} = mode_matrices(R, p(k), xi, rngSeed + 100*k);
    end

    % Build X (requires Tensor Toolbox)
    X = ktensor(lambda(:), A{:});

    % Add noise
    Xfull = full(X);
    N = randn(size(Xfull)); % Random tensor (Gaussian i.i.d. entries)

    if sigma == 0
        sigma = norm(Xfull(:)) / norm(N(:)) * 10^(-SNRdB/20);
        Y = Xfull + sigma*N;
    else
        Y = Xfull + sigma*N;

    end

    % Output tensor
    Y = tensor(Y);

end