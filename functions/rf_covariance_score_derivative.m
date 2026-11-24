function S_pi = rf_covariance_score_derivative(u, Z)
%RF_COVARIANCE_SCORE_DERIVATIVE Average derivative of the VAR covariance score.
% The proof of Proposition 1 (Appendix A.2, p. 17) defines S_pi just before
% the expansion of omega_hat. For one month, its Kronecker formula reads
%   ds_t/dpi' = -(kron(u_t,I_n) + kron(I_n,u_t)) * kron(Z_t',I_n).
% Z_t = Z(:,t) holds the paper's Y_{t-1} in the lag order used by estimate_var.

[n, T] = size(u);
kZ = size(Z, 1);
S_pi = zeros(n^2, n * kZ);

% For regressor j in equation a, column (j-1)*n+a of -kron(Z_t',I_n) is
% du_t/dpi_col = -Z(j,t)*e_a. The product rule for u_t*u_t' gives
%   d(u_t*u_t')/dpi_col = -Z(j,t)*(e_a*u_t' + u_t*e_a').
% reshape(dScore,[],1) stacks the columns (vec) to give this column of ds_t/dpi'.
for t = 1:T
    ut = u(:, t);
    for j = 1:kZ
        Zjt = Z(j, t);
        for a = 1:n
            ea = zeros(n, 1);
            ea(a) = 1;
            dScore = -Zjt * (ea * ut' + ut * ea');
            col = (j - 1) * n + a;
            S_pi(:, col) = S_pi(:, col) + reshape(dScore, [], 1);
        end
    end
end

% This sample mean replaces the expectation on p. 17. Each column is linear
% in sum_t u_t*Z(j,t), which vanishes on the OLS estimation sample.
S_pi = S_pi / T;
end
