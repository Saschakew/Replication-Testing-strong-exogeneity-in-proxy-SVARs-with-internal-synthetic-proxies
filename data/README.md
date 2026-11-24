# Obtain the three input files

Open version 1 of each deposit below and choose **Download this project**.
Sign in to ICPSR and review the download terms for each project. Unzip the
downloads and copy the three files directly into this `data/` folder, keeping
their original names. You can
also open the files individually in the deposit and choose **Download this file**.

| Original replication deposit | File inside the deposit | File beside this README |
|---|---|---|
| [Diego R. Känzig, *The Macroeconomic Effects of Oil Supply News* (V1)](https://doi.org/10.3886/E122886V1) | `Kaenzig_replication/data/dataBaseM.mat` | `dataBaseM.mat` |
| Same Känzig deposit | `Kaenzig_replication/instrument/OilSurprisesMLog.mat` | `OilSurprisesMLog.mat` |
| [Marek Jarociński and Peter Karadi, *Deconstructing Monetary Policy Surprises* (V1)](https://doi.org/10.3886/E231538V1) | `data/data_var/data.csv` | `data.csv` |

The oil application reads the six monthly VAR series from `dataBaseM.mat` and
column 14 of `oilProxiesWTIM` from `OilSurprisesMLog.mat`. The monetary policy
application reads five US VAR series and the `ff4_hf` proxy from `data.csv`.

For Table 2, the code keeps the 60 months from February 1985 through January
1990 and sets their missing `ff4_hf` values to zero. Both applications use
every residual month for the tests and only nonzero surprises for the reported
skewness and excess kurtosis.

If you need to check the data vintage, these are the SHA-256 checksums of the
files used for replication. The two CSV checksums reflect line endings and
give identical observations.

| File | SHA-256 |
|---|---|
| `dataBaseM.mat` | `053dfb7ccc090438cd8ce9aefe9ac33921206ae0d61def26a012fccedd1fe234` |
| `OilSurprisesMLog.mat` | `7903860a446634e6b5e95fdd2b5208ada508cf64184221ccb0f3468e96003287` |
| `data.csv` (LF) | `aff529a13191410bab5d58955c978d6d79de7138f061dd53bb6676d3e8fb4b3e` |
| `data.csv` (CRLF, as downloaded from ICPSR) | `45b6b3e9723ea4937e4723a6e0ee3ed99cb874143049b05fcc1c0b6aedd70489` |
