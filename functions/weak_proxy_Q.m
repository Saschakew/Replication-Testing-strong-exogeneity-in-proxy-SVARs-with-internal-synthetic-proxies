function Q = weak_proxy_Q(beta, cache)
%WEAK_PROXY_Q Compute the weak-test loss at one candidate beta.
% ISP Eq. (13) on p. 8 and the unnumbered SP display before Proposition 6
% on p. 24 both give Q = T*gbar'*S^(-1)*gbar. Here both gbar and S use
% the supplied beta. standard_test holds its weight fixed within each fit.

% 1. Form the raw moments g and the adjusted moments g_perp, one month per column.
% cache.Minv_m(:,t) holds M^(-1)*m_t. The standard functions adjust one month
% at a time. Here each matrix product applies the same formula to all columns.
if strcmp(cache.method, 'SP')
    % Eq. (A.3), p. 21. sp_moments supplies the same z_t^i*r_t as in the standard test.
    [g, h] = sp_moments(beta, cache.u, cache.z, cache.L);
    G_pi = localSpGpi(beta, cache, h);
    % Proposition 5 (p. 22) gives g_perp in sp_optimal_weight. The process
    % just above (T3-W) on p. 24 uses the same VAR correction at this beta.
    g_perp = g - G_pi * cache.Minv_m;
else
    % The stack below Eq. (6), p. 5, uses z_t*r_t first, then (z_t^*)^i*r_t.
    [g, zstar, r, ~, invSigma_u, sigma1] = isp_moments(beta, cache.u, cache.z, cache.L);
    [G_pi, G_omega] = localIspDerivatives(beta, cache, zstar, r, invSigma_u, sigma1);
    % As in isp_optimal_weight, Aadj collects the slope terms in the proof of
    % Proposition 1 (pp. 17-18) and retains the sample S_pi. With S_pi = 0
    % and S_omega = -I the adjustment is -G_pi*M^(-1)*m_t + G_omega*s_t,
    % as in the process immediately above Eq. (13) on p. 8.
    Aadj = G_pi - G_omega * cache.S_omega_inv * cache.S_pi;
    g_perp = g - Aadj * cache.Minv_m - G_omega * cache.S_omega_inv * cache.s;
end

% 2. Average the raw moments and estimate the covariance of the adjusted moments.
% See (T2-W)(iv), p. 16 (ISP), and (T3-W)(iv), p. 24 (SP).
gbar = mean(g, 2);
centered = g_perp - mean(g_perp, 2);
S = (centered * centered') / cache.T;
S = (S + S') / 2;

% 3. Evaluate the paper's quadratic form. An infinite loss skips this beta.
if any(~isfinite(gbar)) || any(~isfinite(S(:)))
    Q = Inf;
    return
end

[~, flag] = chol(S);  % A zero flag means S is positive definite.
if flag ~= 0
    Q = Inf;
else
    Q = cache.T * (gbar' * S^(-1) * gbar);
end
if ~isfinite(Q)
    Q = Inf;
end
end

function G_pi = localSpGpi(beta, cache, h)
% Same G_pi loop as sp_optimal_weight. Proposition 5 (p. 22) gives
% G_pi = -kron(E[h(z_t)*Y_{t-1}'], A_beta), with A_beta from Eq. (A.3) on p. 21.
% Each block below averages -h_i(z_t)*kron(Z_t',A_beta) over the T months.
k = cache.k;
G_pi = zeros(cache.q, cache.d_pi);

e1 = [1; zeros(k, 1)];
Jsel = [zeros(k, 1), eye(k)];
A_beta = Jsel - beta * e1';

for t = 1:cache.T
    kronZ_A = kron(cache.Z(:,t)', A_beta);
    for i = 1:cache.L
        idx = (i-1)*k+1:i*k;
        G_pi(idx,:) = G_pi(idx,:) - h(i,t) * kronZ_A;
    end
end
G_pi = G_pi / cache.T;
end

function [G_pi, G_omega] = localIspDerivatives(beta, cache, zstar, r, invSigma_u, sigma1)
k = cache.k;
n = cache.n;
G_pi = zeros(cache.q, cache.d_pi);
G_omega = zeros(cache.q, n^2);

e1 = [1; zeros(k, 1)];
Jsel = [zeros(k, 1), eye(k)];
A_beta = Jsel - beta * e1';
b = [1; beta];
bInv = b' * invSigma_u;

% These are the same two Jacobians computed in isp_optimal_weight, evaluated at beta.
% Appendix A.5, p. 27, step 4(b) gives G_pi^*. In the first block z_t is
% observed, so only r_t changes and step 4(c) gives a zero covariance derivative.
% In the higher blocks term1 differentiates r_t and term2 differentiates (z_t^*)^i.
for t = 1:cache.T
    Zt = cache.Z(:,t);
    zt = cache.z(t);
    zst = zstar(t);
    rt = r(:,t);
    ut = cache.u(:,t);

    kronZ_A = kron(Zt', A_beta);
    rowZ_bInv = kron(Zt', bInv) / sigma1;

    idx1 = 1:k;
    G_pi(idx1,:) = G_pi(idx1,:) - zt * kronZ_A;

    for i = 2:cache.L
        idx = (i-1)*k+1:i*k;

        term1 = -(zst^i) * kronZ_A;
        term2 = -i * (zst^(i-1)) * (rt * rowZ_bInv);
        G_pi(idx,:) = G_pi(idx,:) + term1 + term2;

        % Step 4(c) on p. 27 gives G_omega^*. Its bracket is inside below.
        % row1 gives the numerator term and rowNorm the scale term from Eq. (6).
        row1 = kron((ut' * invSigma_u), bInv);
        inside = -row1 / sigma1;
        rowNorm = kron(bInv, bInv);
        inside = inside + (zst / (2 * sigma1^2)) * rowNorm;
        G_omega(idx,:) = G_omega(idx,:) + i * (zst^(i-1)) * (rt * inside);
    end
end

G_pi = G_pi / cache.T;
G_omega = G_omega / cache.T;
end
