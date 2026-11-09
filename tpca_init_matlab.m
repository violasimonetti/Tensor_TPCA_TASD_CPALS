function init_result = tpca_init_matlab(X, R)
%TPCA_INIT_MATLAB Tensor PCA initialization for CP-ALS.
%
%   init_result = tpca_init_matlab(X, R)
%
%   This follows the Python TensorPCA.t_pca(R) code:
%       1. Unfold tensor along each mode
%       2. Compute X_n * X_n'
%       3. Take top R eigenvectors
%
%   Input:
%       X : dense array or Tensor Toolbox tensor
%       R : target CP rank
%
%   Output:
%       init_result.U      : cell array of factor matrices
%       init_result.s_hat  : cell array of scale estimates
%
%   Usage:
%       init_result = tpca_init_matlab(Y, R);
%       U0 = init_result.U;
%       P = cp_als(Y, R, 'init', U0);

    % Convert Tensor Toolbox tensor to dense MATLAB array if needed
    if isa(X, 'tensor') || isa(X, 'sptensor') || isa(X, 'ttensor') || isa(X, 'ktensor')
        X_array = double(full(X));
    else
        X_array = X;
    end

    % Check for missing values, like the Python code
    if any(isnan(X_array(:)))
        error('The tensor contains missing values.');
    end

    % Tensor order and shape
    d = ndims(X_array);
    p = size(X_array);

    % Storage
    U = cell(d, 1);
    s_hat = cell(d, 1);

    for mode = 1:d

        % ---------------------------------------------------------------
        % Mode-n unfolding
        %
        % Python:
        % np.moveaxis(tensor, mode, 0).reshape((shape[mode], -1), order='F')
        %
        % MATLAB equivalent:
        % move mode to first dimension, then reshape column-wise.
        % ---------------------------------------------------------------

        Xn = unfold_mode_tpca(X_array, mode);

        % ---------------------------------------------------------------
        % Eigen-decomposition of Xn * Xn'
        %
        % Python:
        % s, gamma = LA.eigh(Xn @ Xn.transpose())
        % ---------------------------------------------------------------

        C = Xn * Xn';

        % Make C exactly symmetric numerically
        C = (C + C') / 2;

        [gamma, D] = eig(C);

        eigvals = diag(D);

        % Sort eigenvalues from largest to smallest
        [eigvals_sorted, idx] = sort(eigvals, 'descend');

        % Keep top R eigenvalues/eigenvectors
        s_hat{mode} = sqrt(max(eigvals_sorted(1:R), 0));

        U{mode} = real(gamma(:, idx(1:R)));
    end

    init_result = struct;
    init_result.U = U;
    init_result.s_hat = s_hat;
end


function Xn = unfold_mode_tpca(X, mode)
%UNFOLD_MODE_TPCA Mode-n unfolding matching the Python TensorPCA code.

    d = ndims(X);
    sz = size(X);

    order = [mode, 1:mode-1, mode+1:d];

    X_perm = permute(X, order);

    Xn = reshape(X_perm, sz(mode), []);
end