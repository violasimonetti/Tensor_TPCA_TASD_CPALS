function result = tasd_init_matlab(X, R)
%TASD_INIT_MATLAB Initialization equivalent to the R h_d_new() function.
%
%   result = tasd_init_matlab(X, R)
%
%   Input:
%       X : dense MATLAB array or Tensor Toolbox tensor-like object
%       R : CP rank
%
%   Output:
%       result.U      : cell array of factor matrices, one per mode
%       result.tensor : dense reconstructed initialization tensor
%
%   Usage with Tensor Toolbox CP_ALS:
%
%       init_result = tasd_init_matlab(X, R);
%       U0 = init_result.U;
%       P = cp_als(tensor(X), R, 'init', U0);

    % Convert Tensor Toolbox object to dense MATLAB array if needed
    if isa(X, 'tensor') || isa(X, 'sptensor') || isa(X, 'ktensor') || isa(X, 'ttensor')
        X_array = double(full(X));
    else
        X_array = X;
    end

    p = size(X_array);
    d = ndims(X_array);

    if d < 3
        error('This initialization is intended for tensors with at least 3 modes.');
    end

    if any(p < R)
        error('Each tensor mode must have size at least R.');
    end

    % ---------------------------------------------------------------------
    % Step 1: HOSVD truncation to multilinear rank (R, R, ..., R)
    % Equivalent to: result_hosvd = hosvd(input_tensor, rep(R, d)) in R
    % ---------------------------------------------------------------------

    Uh = cell(d, 1);

    for k = 1:d
        Xk = unfold_mode(X_array, k);
        [Uk, ~, ~] = svd(Xk, 'econ');
        Uh{k} = Uk(:, 1:R);
    end

    % Core tensor Z = X x_1 Uh{1}' x_2 Uh{2}' ... x_d Uh{d}'
    Z = X_array;
    for k = 1:d
        Z = mode_product(Z, Uh{k}', k);
    end

    % ---------------------------------------------------------------------
    % Step 2: Diagonalization-style initialization on the HOSVD core
    % Equivalent to: result = sim_diag_new(result_hosvd$Z@data, R)
    % ---------------------------------------------------------------------

    if R == 1
        U = Uh;
    else
        sim_result = sim_diag_new_matlab(Z, R);

        U = cell(d, 1);
        for k = 1:d
            U{k} = Uh{k} * sim_result.U{k};
        end
    end

    % ---------------------------------------------------------------------
    % Step 3: Reconstruct the initialization tensor
    % Equivalent to: kruskal_product_high_d(U)
    % ---------------------------------------------------------------------

    result = struct;
    result.U = U;
    result.tensor = kruskal_product_high_d_matlab(U);
end


% ========================================================================
% Equivalent of R function sim_diag_new()
% ========================================================================

function result = sim_diag_new_matlab(X, R)

    p = size(X);
    d = ndims(X);

    % Random unit vectors for modes 3:d
    random_list_1 = cell(d - 2, 1);
    random_list_2 = cell(d - 2, 1);

    for k = 3:d
        a1 = randn(p(k), 1);
        a1 = a1 / norm(a1);
        random_list_1{k - 2} = a1';

        a2 = randn(p(k), 1);
        a2 = a2 / norm(a2);
        random_list_2{k - 2} = a2';
    end

    % Contract tensor along modes 3:d using the two random vector lists
    T1_tensor = X;
    T2_tensor = X;

    for k = 3:d
        T1_tensor = mode_product(T1_tensor, random_list_1{k - 2}, k);
        T2_tensor = mode_product(T2_tensor, random_list_2{k - 2}, k);
    end

    % After contraction, both should be p(1) x p(2) matrices
    T1 = reshape(T1_tensor, p(1), p(2));
    T2 = reshape(T2_tensor, p(1), p(2));

    % U1 = eigenvectors of T1 * pinv(T2)
    % R code:
    % U1 = Re(eigen(T1 %*% MASS::ginv(T2))$vectors[,1:R])
    A = T1 * pinv(T2);

    [V, D] = eig(A);

    % R's eigen() usually returns eigenvalues ordered by decreasing modulus.
    % MATLAB does not guarantee this, so we sort manually.
    eigvals = diag(D);
    [~, idx] = sort(abs(eigvals), 'descend');

    U1 = real(V(:, idx(1:R)));

    % Solve for the remaining factor structure
    % Equivalent to:
    % other_U = MASS::ginv(U1) %*% unfold(input_tensor, 1, 2:d)@data
    X1 = unfold_mode(X, 1);
    other_U = pinv(U1) * X1;

    U = cell(d, 1);
    U{1} = U1;

    for j = 2:d
        U{j} = nan(p(j), R);
    end

    % Recover factors for modes 2:d component by component
    for r = 1:R

        tmp_tensor = reshape(other_U(r, :), p(2:end));

        if d == 3
            % For a 3-way tensor, tmp_tensor is a matrix.
            % R code uses svd(tmp_tensor)$u[,1] and svd(tmp_tensor)$v[,1]
            [Ur, ~, Vr] = svd(tmp_tensor, 'econ');

            U{2}(:, r) = Ur(:, 1);
            U{3}(:, r) = Vr(:, 1);

        else
            % For higher-order tensors, R code calls:
            % cp(as.tensor(tmp_tensor), 1)
            %
            % This requires Tensor Toolbox cp_als.
            Ptmp = cp_als(tensor(tmp_tensor), 1, ...
                          'printitn', 0, ...
                          'maxiters', 50, ...
                          'tol', 1e-10);

            for j = 2:d
                U{j}(:, r) = Ptmp.U{j - 1};
            end
        end
    end

    result = struct;
    result.U = U;
end


% ========================================================================
% Higher-dimensional Kruskal product
% Equivalent to R kruskal_product_high_d(U)
% ========================================================================

function Xhat = kruskal_product_high_d_matlab(U)

    d = numel(U);
    R = size(U{1}, 2);

    dims = zeros(1, d);
    for k = 1:d
        dims(k) = size(U{k}, 1);
    end

    Xhat = zeros(dims);

    for r = 1:R
        vectors = cell(d, 1);
        for k = 1:d
            vectors{k} = U{k}(:, r);
        end

        Xhat = Xhat + outer_product_vectors(vectors);
    end
end


function T = outer_product_vectors(vectors)

    d = numel(vectors);
    dims = zeros(1, d);

    for k = 1:d
        dims(k) = numel(vectors{k});
    end

    T = ones(dims);

    for k = 1:d
        shape = ones(1, d);
        shape(k) = dims(k);

        vk = reshape(vectors{k}, shape);
        T = bsxfun(@times, T, vk);
    end
end


% ========================================================================
% Mode-n unfolding
%
% For tensor X of size I1 x I2 x ... x Id,
% unfold_mode(X,n) returns a matrix of size:
%
%       size(X,n) x prod(size(X,[1:n-1 n+1:d]))
%
% This is the MATLAB analogue of unfold(input_tensor, n, remaining_modes).
% ========================================================================

function Xn = unfold_mode(X, n)

    d = ndims(X);
    order = [n, 1:n-1, n+1:d];

    X_perm = permute(X, order);
    sz = size(X);

    Xn = reshape(X_perm, sz(n), []);
end


% ========================================================================
% Mode-n product
%
% Computes:
%
%       Y = X x_n A
%
% If X is I1 x ... x In x ... x Id
% and A is J x In,
% then Y is I1 x ... x J x ... x Id.
% ========================================================================

function Y = mode_product(X, A, n)

    sz = size(X);
    d = ndims(X);

    In = sz(n);

    if size(A, 2) ~= In
        error('Matrix has incompatible number of columns for mode product.');
    end

    order = [n, 1:n-1, n+1:d];
    X_perm = permute(X, order);

    X_mat = reshape(X_perm, In, []);
    Y_mat = A * X_mat;

    new_sz = sz;
    new_sz(n) = size(A, 1);

    Y_perm = reshape(Y_mat, [new_sz(n), sz(order(2:end))]);

    inverse_order(order) = 1:d;
    Y = permute(Y_perm, inverse_order);
end