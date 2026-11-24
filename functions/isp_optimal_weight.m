function [Wopt, Sout, G_beta] = isp_optimal_weight(betaTilde, u, z, Z, L)
%ISP_OPTIMAL_WEIGHT Compute the efficient weight for the internal synthetic proxy test.

[n, T] = size(u);
kZ = size(Z,1);
d_pi = n * kZ;

[g, zstar, r, Sigma_u, invSigma_u, sigma1] = isp_moments(betaTilde, u, z, L);

% Compare G_beta in the opening display of Proposition 1.
G_beta = betaDerivative(betaTilde, u, z, L, zstar, r, invSigma_u, sigma1);

% Appendix A.5, step 3 gives m_t(pi_hat) and s_t(pi_hat,omega_hat).
m = zeros(d_pi, T);
s = zeros(n^2, T);
for t = 1:T
    ut = u(:,t);
    Zt = Z(:,t);
    m(:,t) = kron(Zt, ut);
    s(:,t) = reshape(ut*ut' - Sigma_u, [], 1);
end

% Appendix A.5, step 4(a) gives the sample Gamma and M = -kron(Gamma,I).
Gamma = (Z*Z')/T;
M = -kron(Gamma, eye(n));

% Near the bottom of p. 5, the paragraph just above the ISP GMM estimator
% defines s_t(pi,omega) = vec(u_t(pi)*u_t(pi)' - Sigma_u(omega)).
% By linearity of vec, this equals vec(u_t*u_t') - omega. At fixed pi
% only the term -omega varies with omega.
% In the proof of Proposition 1 (Appendix A.2, p. 17), the sentence starting
% "For omega_hat_T, denote" defines S_omega = E[ds_t(pi_0,omega_0)/domega'].
% The next sentence states S_omega = -I_(n^2). Its inverse is also -I_(n^2).
S_omega_inv = -eye(n^2);
% For S_pi = E[ds_t/dpi'] in the same paragraph, hold omega fixed and
% differentiate both u_t factors. The helper averages the result over months.
% The OLS normal equations u*Z'/T = 0 give a sample S_pi of zero up to roundoff.
S_pi = rf_covariance_score_derivative(u, Z);

betaTilde = betaTilde(:);
% A_beta is the selector J - beta*iota_1' in Eq. (A.3). The normalized
% impact vector b enters the projection in Eq. (6).
e1 = [1; zeros(n-1,1)];
Jsel = [zeros(n-1,1), eye(n-1)];
A_beta = (Jsel - betaTilde * e1');
b = [1; betaTilde];

bInv = (b' * invSigma_u);

% Appendix A.5 (p. 27), steps 4(b) and 4(c), estimate the two Jacobians
% G_pi^* and G_omega^* in Proposition 1 (p. 6). Each proxy moment in step 3
% occupies n-1 rows. Columns correspond to the d_pi VAR coefficients or
% the n^2 entries of omega = vec(Sigma_u). Both partials hold betaTilde fixed.
G_pi = zeros(L*(n-1), d_pi);
G_omega = zeros(L*(n-1), n^2);
for t = 1:T
    % Zt holds the paper's regressors Y_{t-1} in the VAR's lag order.
    % In step 3, zt is the observed z_t and zst is the fitted internal
    % proxy z_hat_t^*. Here ut is u_t(pi_hat) and rt = A_beta*ut.
    Zt = Z(:,t);
    zt = z(t);
    zst = zstar(t);
    rt = r(:,t);
    ut = u(:,t);

    % From the VAR on p. 5, du_t/dpi' = -kron(Zt',I_n). Multiplying by
    % A_beta = J - betaTilde*iota_1' gives dr_t/dpi' = -kronZ_A.
    kronZ_A = kron(Zt', A_beta);

    % The first block in 4(b) differentiates zt*rt, giving -zt*kronZ_A.
    % At fixed pi neither factor depends on omega, giving the zero first
    % block printed in 4(c). Only G_pi is updated for these rows.
    idx1 = 1:(n-1);
    G_pi(idx1,:) = G_pi(idx1,:) - zt * kronZ_A;

    for i = 2:L
        % These are the n-1 rows for the step 3 moment g_i,t^* = zst^i*rt.
        idx = (i-1)*(n-1)+1:i*(n-1);

        % Step 4(b), first line of the display for i >= 2. The product
        % rule gives zst^i times dr_t/dpi', with omega held fixed.
        term1 = -(zst^i) * kronZ_A;

        % The second line differentiates zst^i. In step 3, zst = bInv*ut/sigma1
        % with bInv = b'*invSigma_u and sigma1 the dotted scale in Eq. (6).
        % At fixed omega, dzst/dpi' = -rowZ_bInv (proof of Lemma 4, p. 19).
        % The column rt times that row gives the outer product in term2.
        % The factor i*zst^(i-1) comes from differentiating the power.
        rowZ_bInv = kron(Zt', bInv) / sigma1;
        term2 = -i * (zst^(i-1)) * (rt * rowZ_bInv);

        G_pi(idx,:) = G_pi(idx,:) + (term1 + term2);

        % Step 4(c), bracket in the last display. Let P = invSigma_u denote
        % the fitted inverse covariance. With N = b'*P*ut and squared scale
        % s^2 = b'*P*b, step 3 gives zst = N/s, where s = sigma1.
        % Since dP = -P*(dSigma_u)*P, dN/domega' = -row1 and
        % d(s^2)/domega' = -rowNorm (see the proof of Lemma 3, p. 18).
        row1 = kron((ut' * invSigma_u), bInv);
        rowNorm = kron(bInv, bInv);
        % inside is dzst/domega'. In the quotient N/s, d(1/s) = -d(s^2)/(2*s^3),
        % so the squared-scale derivative contributes +zst*rowNorm/(2*s^2).
        inside = -row1 / sigma1 + (zst / (2 * sigma1^2)) * rowNorm;
        % At fixed pi, rt is constant. Differentiating zst^i supplies
        % i*zst^(i-1), and the column rt multiplies the bracket on the left.
        G_omega(idx,:) = G_omega(idx,:) + i * (zst^(i-1)) * (rt * inside);
    end
end

% Each update above adds one month's derivative. Division by T gives the
% sample averages G_hat_{pi,T}^* in 4(b) and G_hat_{omega,T}^* in 4(c).
G_pi = G_pi / T;
G_omega = G_omega / T;

% Proof of Proposition 1, Appendix A.2 (p. 17). Write v_t = M^{-1}*m_t.
% Month t contributes -v_t to the expansion of pi_hat. The first line for
% omega_hat gives -S_omega_inv*(s_t - S_pi*v_t) for the same month.
% The first display on p. 18 weights these contributions by G_pi and G_omega.
% Substituting the p. 17 expressions before setting S_pi = 0 gives
%   g_t - G_pi*v_t - G_omega*S_omega_inv*(s_t - S_pi*v_t).
% Collecting the two terms in v_t gives -(G_pi - G_omega*S_omega_inv*S_pi)*v_t.
% Aadj is the MATLAB name for the matrix in parentheses.
Aadj = G_pi - G_omega * S_omega_inv * S_pi;

% At full numerical rank, pinv(M) gives the sample inverse M_hat_T^{-1}.
Minv = pinv(M);

% On p. 17, E[u_t*Y_{t-1}'] = 0 gives S_pi = 0. On the OLS estimation
% sample, u*Z'/T = 0 in exact arithmetic, so the sample Aadj reduces to G_pi.
% Since S_omega_inv = -I, the last subtraction adds G_omega*s(:,t). Thus
%   g_perp(:,t) = g(:,t) - G_pi*(Minv*m(:,t)) + G_omega*s(:,t)
% up to roundoff. This is the plug-in formula below Proposition 1 (p. 6),
% also printed as the second display of Appendix A.5 (p. 27).
g_perp = zeros(size(g,1), T);
for t = 1:T
    g_perp(:,t) = g(:,t) - Aadj * (Minv * m(:,t)) - (G_omega * S_omega_inv * s(:,t));
end

% Sout estimates S_perp^* in the opening display of Appendix A.5. Wopt
% is its inverse, the efficient weight in step 5 and Proposition 1.
Sout = (g_perp * g_perp') / T;
Wopt = pinv(Sout);
end

function Gbeta = betaDerivative(beta, u, z, L, zstar, r, invSigma_u, sigma1)
% Proposition 1 (p. 6, first display) defines G_beta^* as E[dg_t^(Z*)/dbeta'].
% Evaluate the derivative at the supplied beta, holding the fitted VAR fixed.
[n, T] = size(u);
p = n - 1;
Gbeta = zeros(L * p, p);

% The stack below Eq. (6) on p. 5 starts with z_t*r_t. Eq. (A.3) on
% p. 21 defines r_t. The opening sentence of the proof of Lemma 6 (p. 20)
% gives dr_t = -u_1t*d_beta, so this block contributes -z_t*u_1t*I.
idx1 = 1:p;
Gbeta(idx1, :) = -sum(z .* u(1, :)) * eye(p);

% The normalization below Eq. (1) on p. 3 fixes the first entry of
% b = [1; beta] at one. Db is its derivative with respect to beta'.
Db = [zeros(1, p); eye(p)];
% In Eq. (6) on p. 5, rawDer differentiates the numerator b'*invSigma_u*u_t.
% scaleDer is half the gradient of the squared scale b'*invSigma_u*b.
rawDer = Db' * invSigma_u * u;
scaleDer = Db' * invSigma_u * [1; beta];
% Proof of Lemma 6 (p. 20), second line of the first display for dz_t^*.
% sigma1 is the dotted scale in Eq. (6). Column t is the gradient for month t.
dzstar = rawDer / sigma1 - (scaleDer / (sigma1 ^ 2)) * zstar;

for t = 1:T
    rt = r(:, t);
    u1t = u(1, t);
    dzrow = dzstar(:, t)';
    for ell = 2:L
        idx = (ell - 1) * p + 1:ell * p;
        % In the proof of Lemma 6 (p. 21), the first equality after "Therefore,"
        % applies the product rule at beta_0. Write x = zstar(t) and i = ell.
        % With h_i(x) = x^ell (p. 5, below Eq. (6)), the paper's two terms are
        %   h_i'(x)*r_t*(dz_t^*/dbeta') = ell*x^(ell-1)*(rt*dzrow)
        %   h_i(x)*(dr_t/dbeta')        = -x^ell*u1t*eye(p)
        % rt*dzrow is the residual column times the proxy's derivative row.
        % Eq. (A.3) gives r_t = u_2t - beta*u1t, so dr_t/dbeta' = -u1t*eye(p).
        % The next equality in the proof expands rt*dzrow in shocks at beta_0.
        % MATLAB adds the paper's second term first and uses the supplied beta.
        Gbeta(idx, :) = Gbeta(idx, :) ...
            - (zstar(t) ^ ell) * u1t * eye(p) ...
            + ell * (zstar(t) ^ (ell - 1)) * (rt * dzrow);
    end
end

% The sample mean below the stack on p. 5 supplies the 1/T in this
% estimate of G_beta^* (Proposition 1, p. 6).
Gbeta = Gbeta / T;
end
