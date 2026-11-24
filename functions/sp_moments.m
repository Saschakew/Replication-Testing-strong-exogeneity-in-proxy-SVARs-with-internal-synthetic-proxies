function [g, h] = sp_moments(beta, u, z, L)
%SP_MOMENTS Form moment conditions from powers of the observed proxy.


[n, T] = size(u);
% Defined before Eq. (3)
r = u(2:end,:) - beta * u(1,:);
h = zeros(L, T);
g = zeros(L*(n-1), T);

% Compare Eq. (3) for the quadratic case and Eq. (A.3) for all powers.
for i = 1:L
    h(i,:) = z.^i;
    idx = (i-1)*(n-1)+1:i*(n-1);
    g(idx,:) = h(i,:) .* r;
end
end
