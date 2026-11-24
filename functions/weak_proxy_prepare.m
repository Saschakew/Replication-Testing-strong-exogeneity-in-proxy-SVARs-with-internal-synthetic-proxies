function cache = weak_proxy_prepare(u, z, Z, L, method)
%WEAK_PROXY_PREPARE Quantities shared by all candidates in a weak test.
% Q_T^*(beta) in Eq. (13) and Q_T(beta) just before Proposition 6 use the
% same fitted VAR at every candidate. Save its monthly scores for weak_proxy_Q.
[n, T] = size(u);
k = n - 1;
dPi = n * size(Z, 1);
cache = struct('u', u, 'z', z, 'Z', Z, 'n', n, 'T', T, ...
    'k', k, 'q', L*k, 'L', L, 'd_pi', dPi, 'method', method);

% Appendix A.5, step 3 gives m_t = kron(Y_{t-1},u_t). Each column of m
% uses this score, with Z in the lag order of estimate_var.
m = zeros(dPi, T);
for t = 1:T
    m(:,t) = kron(Z(:,t), u(:,t));
end
% Appendix A.5, step 4(a) gives Gamma_hat and M_hat. Minv_m stores
% M_hat^{-1}*m_t. The moment adjustment applies its minus sign in weak_proxy_Q.
Gamma = (Z * Z') / T;
M = -kron(Gamma, eye(n));
cache.Minv_m = pinv(M) * m;

% For ISP, Appendix A.5, steps 1 and 3 give Sigma_hat and the covariance
% scores s_t. The proof of Proposition 1 in Appendix A.2 defines their
% derivatives S_pi and S_omega = -I.
if strcmp(method, 'ISP')
    Sigma_u = (u * u') / T;
    cache.s = zeros(n^2, T);
    for t = 1:T
        ut = u(:,t);
        cache.s(:,t) = reshape(ut * ut' - Sigma_u, [], 1);
    end
    cache.S_pi = rf_covariance_score_derivative(u, Z);
    cache.S_omega_inv = -eye(n^2);
end
end
