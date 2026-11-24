function [beta, J] = standard_test(u, z, Z, betaStart, L, method)
%STANDARD_TEST Two-step efficient GMM and its overidentification statistic.
% beta estimates the impact ratios in b(beta), defined just after Eq. (1).
% The ISP minimizer is the unnumbered beta_hat_T^*(W) just before Eq. (7).
% Eq. (A.4) defines the SP minimizer beta_hat_T(W) and its mean moment gbar.

if strcmp(method, 'ISP')
    moments = @(b) isp_moments(b, u, z, L);
    weight = @(b) isp_optimal_weight(b, u, z, Z, L);
else
    moments = @(b) sp_moments(b, u, z, L);
    weight = @(b) sp_optimal_weight(b, u, z, Z, L);
end
solver = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
    'HessianApproximation', 'lbfgs', 'TolFun', 1e-8, 'TolX', 1e-8, ...
    'MaxIter', 2000, 'Display', 'off');
penalty = 1e20;

% Appendix A.5, step 2 illustrates a preliminary ISP fit with identity or
% naive weighting. These applications use their method's efficient W at
% the external-proxy ratios for the first fit, holding W fixed in that fit.
[W, S, Gbeta] = weight(betaStart);
checkWeight(W, S, Gbeta);
[preliminary, loss, flag] = fminunc(@(b) criterion(b, W), betaStart, solver);
checkFit(preliminary, loss, flag);

% Update W at the preliminary fit and estimate beta again. The efficient
% weights are the inverse of S_perp^* (ISP, Appendix A.5, step 5) and
% the inverse of S_perp (SP, Corollary 1).
[W, S, Gbeta] = weight(preliminary);
checkWeight(W, S, Gbeta);
[beta, loss, flag] = fminunc(@(b) criterion(b, W), preliminary, solver);
checkFit(beta, loss, flag);
% Eq. (7) gives J_T^*(W) for ISP and Eq. (A.5) gives J_T(W) for SP.
% Both multiply the fitted quadratic form gbar' * W * gbar by T months.
gbar = mean(moments(beta), 2);
loss = real(gbar' * W * gbar);
assert(isfinite(loss) && loss >= 0, 'Invalid final GMM criterion.');
J = size(u, 2) * loss;

    function value = criterion(b, weightMatrix)
        % At each trial b, form the quadratic loss defining beta_hat_T^*(W)
        % just before Eq. (7), or beta_hat_T(W) in Eq. (A.4). Invalid trials
        % receive a finite penalty so the optimizer can continue.
        try
            meanMoment = mean(moments(b), 2);
            value = real(meanMoment' * weightMatrix * meanMoment);
        catch
            value = penalty;
        end
        if ~isfinite(value)
            value = penalty;
        end
    end
end

function checkFit(beta, loss, flag)
assert(flag > 0 && isreal(beta) && all(isfinite(beta)) && ...
    isfinite(loss) && loss < 1e19, 'GMM optimization failed to converge.');
end

function checkWeight(W, S, Gbeta)
% Proposition 1 (ISP) and Corollary 1 (SP) use the inverse moment covariance.
% pinv(S) can return a finite weight even if S is singular.
assert(rank(S) == size(S, 1) && all(isfinite(W(:))), ...
    'The moment covariance must have full rank and its inverse must be finite.');
[~, flag] = chol((W + W') / 2);
assert(flag == 0, 'The GMM weight must be positive definite.');
% Assumptions (T2)(v) for ISP and (T3)(iv) for SP require identification.
assert(rank(Gbeta) == size(Gbeta, 2), 'The moment derivative must have full column rank.');
end
