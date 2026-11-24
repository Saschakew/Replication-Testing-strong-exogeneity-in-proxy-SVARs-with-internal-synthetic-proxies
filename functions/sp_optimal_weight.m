function [Wopt, Sout, G_beta] = sp_optimal_weight(betaTilde, u, z, Z, L)
%SP_OPTIMAL_WEIGHT Compute the efficient weight for the synthetic proxy test.


[n, T] = size(u);
kZ = size(Z,1);
d_pi = n * kZ;

[g, h] = sp_moments(betaTilde, u, z, L);

% Differentiating Eq. (A.3) gives the sample derivative -mean(h_i*u_1)*I.
% Its population counterpart in Proposition 5 is G_beta = -kron(rho,I).
rho_hat = mean(h .* u(1,:), 2);
G_beta = -kron(rho_hat, eye(n - 1));

% m(:,t) is the score m_t(pi_hat) in the unnumbered Section 3 normal
% equation. Z(:,t) contains the regressors Y_{t-1} in the code's lag order.
m = zeros(d_pi, T);
for t = 1:T
    m(:,t) = kron(Z(:,t), u(:,t));
end

% Proposition 5 defines Gamma = E[Y_{t-1}Y_{t-1}'] and M = -kron(Gamma,I).
Gamma = (Z*Z')/T;
M = -kron(Gamma, eye(n));
Minv = pinv(M);

betaTilde = betaTilde(:);
e1 = [1; zeros(n-1,1)];
Jsel = [zeros(n-1,1), eye(n-1)];
A_beta = (Jsel - betaTilde * e1');

% Proposition 5 gives G_pi = -kron(E[h(z_t)*Y_{t-1}'], A_beta), where
% A_beta is J - beta*iota_1' in Eq. (A.3). 
G_pi = zeros(L*(n-1), d_pi);
for t = 1:T
    kronZ_A = kron(Z(:,t)', A_beta);
    for i = 1:L
        idx = (i-1)*(n-1)+1:i*(n-1);
        G_pi(idx,:) = G_pi(idx,:) - h(i,t) * kronZ_A;
    end
end
G_pi = G_pi / T;

% g_perp(:,t) is the sample version of g_t^{(Z),perp} in Proposition 5.
% Subtracting G_pi*M^{-1}*m_t accounts for fitting the VAR.
g_perp = zeros(size(g,1), T);
for t = 1:T
    g_perp(:,t) = g(:,t) - G_pi * (Minv * m(:,t));
end

% Sout estimates S_perp in Proposition 5 by the mean outer product.
% Wopt implements the efficient weight S_perp^{-1} in Corollary 1.
Sout = (g_perp * g_perp') / T;
Wopt = pinv(Sout);

end
