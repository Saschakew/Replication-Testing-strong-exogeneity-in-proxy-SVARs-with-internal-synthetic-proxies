function [Qupper, decision, reject] = weak_test(u, z, Z, betaStart, standardBeta, L, method, cutoff)
%WEAK_TEST Find the lowest weak-test loss Q and compare it with the cutoff.
% Eq. (15) on p. 8 (ISP) and Eq. (A.6) on p. 25 (SP) keep each beta in B
% whose loss is at most the cutoff. One such beta proves the set is nonempty.

% 1. Build the starting values. Each column of startValues is one beta.
% Below Eq. (1) on p. 3, b(beta) = (1,beta')', so there are k = n-1 ratios.
% The paragraph after Proposition 4 (p. 8) suggests multiple starts on compact
% B. These applications use [-8,8] for each ratio.
k = numel(betaStart);
lower = -8 * ones(k,1);
upper =  8 * ones(k,1);
startValues = makeStartValues(u, z, betaStart, standardBeta, lower, upper);

% 2. Compute the loss at every start and save the smallest one.
% weak_proxy_Q evaluates Eq. (13) on p. 8 for ISP or the unnumbered display
% before Proposition 6 on p. 24 for SP. Both have the form T*gbar'*S^(-1)*gbar,
% with mean moments gbar and adjusted covariance S evaluated at the same beta.
% The monthly VAR residuals u, proxy z and lag regressors Z are fixed.
cache = weak_proxy_prepare(u, z, Z, L, method);
startLoss = zeros(size(startValues,2), 1);
for j = 1:numel(startLoss)
    startLoss(j) = weak_proxy_Q(startValues(:,j), cache);
end
Qupper = min(startLoss);

% weak_proxy_Q returns Inf when it cannot evaluate a candidate.
if ~isfinite(Qupper)
    Qupper = NaN;
    decision = 'numerical_failure';
    reject = false;
    return
end

% 3. Refine up to four starts with the lowest finite losses.
% After each local search, keep the smaller of its best loss and Qupper.
[~, order] = sort(startLoss, 'ascend');
nRefine = min(4, sum(isfinite(startLoss)));
options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point', ...
    'MaxFunctionEvaluations', 150, 'MaxIterations', 100, ...
    'StepTolerance', 1e-6, 'OptimalityTolerance', 1e-8);
for j = 1:nRefine
    start = startValues(:,order(j));
    refinedLoss = refineStart(start, cache, lower, upper, options);
    Qupper = min(Qupper, refinedLoss);
end

% 4. Apply the empty-set rule in Eqs. (15) and (A.6).
% Qupper bounds the infimum over B from above. Its chi-square upper tail gives
% the p-value lower bound in Tables 1 and 2. Propositions 3 (p. 8) and 6 (p. 24)
% give the pointwise null law at the true beta with L*(n-1) degrees of freedom.
% Propositions 4 (p. 8) and 7 (p. 25) give size control for the exact rule.
% Here 'empty' is a numerical decision. Allow a small tolerance above the cutoff.
tolerance = 1e-8 * max(1, abs(cutoff));
reject = Qupper > cutoff + tolerance;
if reject
    decision = 'empty';
else
    decision = 'nonempty';
end
end

function startValues = makeStartValues(u, z, betaStart, standardBeta, lower, upper)
% Start at zero, the driver's proxy ratios, and the method's two-step GMM fit.
k = numel(betaStart);
startValues = [zeros(k,1), betaStart, standardBeta];

% The display after Assumption (A1) on p. 3 gives beta_0 = E[z*u_2]/E[z*u_1].
% Add the sample ratio if the denominator is large enough. It can repeat betaStart.
denominator = mean(z .* u(1,:));
if isfinite(denominator) && abs(denominator) > 1e-8
    ratio = mean(u(2:end,:) .* z, 2) / denominator;
    startValues(:,end+1) = ratio;
end

% Try +4 and -4 on each axis in turn, then 32 Sobol points across the box.
for j = 1:k
    step = zeros(k,1);
    step(j) = (upper(j) - lower(j)) / 4;
    startValues = [startValues, step, -step]; %#ok<AGROW>
end
startValues = min(max(startValues, lower), upper);
sobol = net(sobolset(k, 'Skip', 2), 32)';
startValues = [startValues, lower + (upper - lower) .* sobol];

% Use rounded values to find duplicates and keep the original column for scoring.
startValues = startValues(:, all(isfinite(startValues), 1));
rounded = round(startValues' / 1e-10) * 1e-10;
[~, first] = unique(rounded, 'rows', 'stable');
startValues = startValues(:,first);
end

function bestLoss = refineStart(start, cache, lower, upper, options)
% fmincon also probes nearby points to estimate derivatives. Any trial inside
% the box can improve the reported bound, including one the solver steps past.
bestLoss = Inf;
fmincon(@trialLoss, start, [], [], [], [], lower, upper, [], options);

    function loss = trialLoss(beta)
        beta = min(max(beta(:), lower), upper);
        loss = weak_proxy_Q(beta, cache);
        bestLoss = min(bestLoss, loss);
    end
end
