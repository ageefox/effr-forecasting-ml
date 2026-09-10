# Data source

`effr_monthly.csv` contains monthly averages of the effective federal funds rate from July 1954 through February 2017.

- Publisher: Board of Governors of the Federal Reserve System
- Release: H.15 Selected Interest Rates
- Series: `H15/H15/RIFSPFF_N.M`
- Description: Federal funds effective rate
- Unit: percent per year
- Frequency: monthly average
- Retrieved: September 10, 2026
- [Official series preview](https://www.federalreserve.gov/datadownload/Preview.aspx?pi=400&preview=H15%2FH15%2FRIFSPFF_N.M&rel=H15)
- [Data Download Program](https://www.federalreserve.gov/datadownload/Download.aspx?rel=H15)

The snapshot uses the first calendar day as the timestamp for each source month and retains the source value to two decimal places. It is intentionally capped at February 2017 to preserve the original study period. The repository's `run.json` records the exact file hash used for the published benchmark.

The Board states that information on its website is in the public domain unless otherwise indicated and asks users to cite the Board as the source. See the [Federal Reserve Board disclaimer](https://www.federalreserve.gov/disclaimer.htm). The repository's MIT license applies to the code; this data file retains its source attribution and is provided under the source's terms.
