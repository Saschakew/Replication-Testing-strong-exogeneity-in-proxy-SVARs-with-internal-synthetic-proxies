function [g, zstar, r, Sigma_u, invSigma_u, sigma1] = isp_moments(beta, u, z, L)
%ISP_MOMENTS Form moments from the external proxy and a candidate internal proxy.
% Each column of g corresponds to one residual month.

[n, T] = size(u);
k = n - 1;
% Defined before Eq. (3)
r = u(2:end,:) - beta * u(1,:);
% Defined after Eq. (1)
Sigma_u = (u * u') / T;
invSigma_u = pinv(Sigma_u);

% Section 2 defines b(beta) just after Eq. (1) by fixing its first entry at one. 
b = [1; beta];
% Eq. (6) gives the formula evaluated here.
bInv = b' * invSigma_u;
scaleSquared = bInv * b;
assert(isfinite(scaleSquared) && scaleSquared > 0, 'Invalid internal-proxy variance.');
sigma1 = sqrt(scaleSquared);
zstar = (bInv * u) / sigma1;

% The unnumbered stack g_t^(Z*) between Eq. (6) and Eq. (7) is written
% out in Appendix A.5, step 3. Its first block uses z*r. The remaining
% blocks use zstar^i*r, with i = 2, 3 in the applications.
g = zeros(L * k, T);
g(1:k,:) = z .* r;
for i = 2:L
    rows = (i - 1) * k + (1:k);
    g(rows,:) = (zstar.^i) .* r;
end
end
