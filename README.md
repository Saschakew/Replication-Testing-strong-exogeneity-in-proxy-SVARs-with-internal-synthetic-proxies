# Replicating the two benchmark applications

This package reproduces the oil supply news and monetary policy applications in
Section 6 of *Testing strong exogeneity in proxy SVARs with internal synthetic
proxies* by Sascha A. Keweloh and Shu Wang. It generates both exogeneity-test
tables and the skewness and excess kurtosis quoted in the text.
Equation numbers in the code follow the [published article](https://doi.org/10.1016/j.jeconom.2026.106354).

## Run in MATLAB

1. Clone the repository or extract the ZIP. Follow the [data instructions](data/README.md)
   to download the three original files and place them in `data/`.
2. Use MATLAB with Optimization Toolbox and Statistics and Machine Learning Toolbox.
   The results were checked with MATLAB R2026a Update 4 on Windows.
3. Set MATLAB’s current folder to the one containing `run_replication.m` and run

   ```matlab
   run_replication;
   ```

The command estimates both applications and writes three files in `output/`.
Running it again replaces those files.

| Output | Section 6 result |
|---|---|
| `tab_kaenzig_exogeneity_tests.tex` | Four tests for the oil supply news proxy |
| `tab_jk_exogeneity_tests.tex` | Four tests for the monetary policy proxy |
| `proxy_moments.csv` | Test dates, sample counts, skewness and excess kurtosis for both proxies |

To inspect the unrounded results in MATLAB, use
`[oilTests, monetaryTests, moments] = run_replication;` in step 3.
The TeX files contain tabular environments that use the `booktabs` package.

## Reading the code

Start with the application of interest. Each driver selects its sample, fits
the VAR and aligns the proxy to residual months.
[Oil driver](applications/application_kaenzig.m) · [Monetary policy driver](applications/application_jk.m)

Both drivers call the same four-row table calculation. The standard test
estimates the impact ratios twice, updating the efficient weight between
estimates. In these applications the weak test searches from −8 to 8 for
each impact ratio and refines the four best starting points.
[Table calculation](functions/exogeneity_tests.m) · [Standard test](functions/standard_test.m) · [Weak test](functions/weak_test.m)
