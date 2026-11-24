function [residuals, Z] = estimate_var(observations, lags)
%ESTIMATE_VAR Fit the monthly VAR with an intercept by least squares.
% Each row of observations is one month. Row one of residuals belongs to
% month lags + 1, the first month with a complete lag history.

[months, variables] = size(observations);
Y = observations(lags + 1:end, :);
X = ones(months - lags, 1 + variables * lags);
% Y holds the dependent vectors y_t of Eq. (1). Each row of X has
% the entries of the paper's Y_{t-1} (Section 3, opening paragraph).
% The paper lists lags newest first. X lists them oldest first and uses
% that order for the coefficients and the regressors returned in Z.
for lag = 1:lags
    columns = 1 + (lags - lag) * variables + (1:variables);
    X(:, columns) = observations(lags + 1 - lag:end - lag, :);
end
% This solves the unnumbered least-squares normal equation for pi_hat
% in Section 3. Row t of residuals is the fitted innovation u_t(pi_hat).
coefficients = X \ Y;
residuals = Y - X * coefficients;
Z = X';  % The inference routines use one observation per column.
end
