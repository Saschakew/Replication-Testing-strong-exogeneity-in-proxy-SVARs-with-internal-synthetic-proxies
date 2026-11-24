function [tests, proxy, testDates] = application_kaenzig(dataDir)
%APPLICATION_KAENZIG Test the oil supply news proxy in the six-variable VAR.
% Section 6.1 gives the oil VAR and proxy, and reports the four tests in
% Table 1. Here lags is the VAR order p in Eq. (1), and momentBlocks is L.

lags = 12;
momentBlocks = 3;
alpha = 0.10;

% The data columns are the real oil price, world oil production, world oil
% inventories, world industrial production, US industrial production and US CPI.
monthly = load(fullfile(dataDir, 'dataBaseM.mat'), 'data', 'sampleDates');
dates = datetime(monthly.sampleDates, 'InputFormat', "yyyy'M'MM");
inSample = dates >= datetime(1974, 1, 1) & dates <= datetime(2017, 12, 1);
observations = monthly.data(inSample, :);
dates = dates(inSample);

[residuals, regressors] = estimate_var(observations, lags);
testDates = dates(lags + 1:end);
% Twelve lags put the first residual in January 1975. Match by calendar
% month across the two files. Column 14 is the first principal component of
% oil futures price changes around OPEC announcements, using maturities
% spanning the first year.
oil = load(fullfile(dataDir, 'OilSurprisesMLog.mat'), 'oilProxiesWTIM', 'sampleDatesProxy');
proxyDates = datetime(oil.sampleDatesProxy, 'InputFormat', "yyyy'M'MM");
[covered, proxyIndex] = ismember(testDates, proxyDates);
assert(all(covered), 'An oil VAR residual month is missing from the proxy file.');
proxy = oil.oilProxiesWTIM(proxyIndex, 14)';

% thetaStart is the sample counterpart of the unnumbered beta_0 ratio
% immediately after Assumption (A1) in Section 2. Dividing each mean(z*u_j)
% by mean(z*u_1) measures impacts per unit impact on the real oil price.
thetaStart = (mean(proxy' .* residuals(:, 2:end)) ./ mean(proxy' .* residuals(:, 1)))';
tests = exogeneity_tests(residuals', proxy, regressors, ...
    thetaStart, momentBlocks, alpha);
end
