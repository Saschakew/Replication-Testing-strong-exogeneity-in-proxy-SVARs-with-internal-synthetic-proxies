function [tests, proxy, testDates] = application_jk(dataDir)
%APPLICATION_JK Test the monetary policy proxy in the five-variable US VAR.
% Section 6.2 gives the monetary VAR and proxy, and reports the four tests
% in Table 2. Here lags is the VAR order p in Eq. (1), and momentBlocks is L.

lags = 12;
momentBlocks = 3;
alpha = 0.10;

monthly = readtable(fullfile(dataDir, 'data.csv'));
dates = datetime(monthly.year, monthly.month, 1);
inSample = dates >= datetime(1984, 2, 1) & dates <= datetime(2016, 12, 1);
% The columns are the one-year Treasury yield, log S&P 500 index, log real GDP,
% log GDP deflator and excess bond premium.
variables = {'gs1', 'logsp500', 'us_rgdp', 'us_gdpdef', 'ebpnew'};
observations = monthly{inSample, variables};
dates = dates(inSample);

[residuals, regressors] = estimate_var(observations, lags);
testDates = dates(lags + 1:end);
% ff4_hf measures the change in three-month-ahead federal funds futures
% around FOMC announcements. Its coverage starts in February 1990.
surprises = monthly.ff4_hf(inSample);
proxy = surprises(lags + 1:end)';
% To reproduce Table 2, retain the 60 months from February 1985 through
% January 1990 and code their missing surprises as zero. Checking the dates
% first catches any gap within proxy coverage.
missing = isnan(proxy);
assert(isequal(missing, testDates' < datetime(1990, 2, 1)), ...
    'The ff4_hf coverage differs from the published data.');
proxy(missing) = 0;

% thetaStart is the sample counterpart of the unnumbered beta_0 ratio
% immediately after Assumption (A1) in Section 2. Dividing each mean(z*u_j)
% by mean(z*u_1) measures impacts per unit impact on the one-year yield.
thetaStart = (mean(proxy' .* residuals(:, 2:end)) ./ mean(proxy' .* residuals(:, 1)))';
tests = exogeneity_tests(residuals', proxy, regressors, ...
    thetaStart, momentBlocks, alpha);
end
