function results = exogeneity_tests(u, z, Z, betaStart, L, alpha)
%EXOGENEITY_TESTS Compute the four rows of a Section 6 table.
% u contains VAR residuals, z the external proxy, and Z the
% lagged VAR regressors.  The synthetic proxy (SP) test adds powers of
% z. The internal synthetic proxy (ISP) test adds powers of the candidate
% shock constructed from u. Tables 1 and 2 use three moment blocks. The
% code also reports rejection decisions at the 10% level.

assert(size(u,2) == numel(z) && size(Z,2) == numel(z) && ...
    all(isfinite(u(:))) && all(isfinite(z)) && all(isfinite(Z(:))) && ...
    all(isfinite(betaStart)), 'The test inputs must be finite and aligned by month.');
methods = {'ISP'; 'SP'};
parameters = numel(betaStart);
% With k = n-1 fitted ratios, J has (L-1)*k degrees of freedom under
% Proposition 1 (ISP) and Corollary 1 (SP). At the true candidate the
% weak-test Q has L*k degrees of freedom under Proposition 3 (ISP) and
% Proposition 6 (SP). The cutoff uses that pointwise law in each weak row.
df = [(L - 1) * parameters * ones(2,1); L * parameters * ones(2,1)];
critical_value = chi2inv(1 - alpha, df);
statistic = zeros(4,1);
decision = repmat({''}, 4, 1);
reject = false(4,1);
standardBeta = cell(2,1);

% The standard estimates also supply starting points for the weak tests.
for i = 1:2
    [standardBeta{i}, statistic(i)] = standard_test(u, z, Z, betaStart, L, methods{i});
end
for i = 1:2
    row = i + 2;
    [statistic(row), decision{row}, reject(row)] = weak_test( ...
        u, z, Z, betaStart, standardBeta{i}, L, methods{i}, critical_value(row));
end

% Eq. (15) for ISP and Eq. (A.6) for SP invert Q over the candidate set.
% The smallest Q found bounds the infimum from above. Its decreasing
% chi-square upper tail therefore bounds the corresponding p-value from below.
p_value = chi2cdf(statistic, df, 'upper');
reject(1:2) = p_value(1:2) <= alpha;
test = [methods; methods];
version = {'Standard'; 'Standard'; 'Weak robust'; 'Weak robust'};
statistic_name = {'J'; 'J'; 'Q'; 'Q'};
results = table(test, version, statistic_name, statistic, p_value, reject, ...
    critical_value, decision, df);
end
