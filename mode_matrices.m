function Ak = mode_matrices(R, p, xi, rngSeed)
% This function computes mode-k matrix for CP tensor decomposition.
%
% Inputs:
% - R: (scalar) rank, it must be R < p otherwise this construction does not work
% - p: (scalar) dimension of Ak
% - xi: (scalar) coherence parameter
% - rngSeed: (scalar) seed for reprudicibility
%
% Outputs:
% - Ak: (p x R) mode-k matrix

    rng(rngSeed);

    % Random orthonormal basis Q (p x R)
    [Q,~] = qr(randn(p,R),0);

    % Target Gram matrix
    G = (1-xi)*eye(R) + xi*ones(R);

    % Factor with desired Gram
    S = chol(G,'upper');
    Ak = Q*S;
   
end