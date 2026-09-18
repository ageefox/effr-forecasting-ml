# Data sources

The two CSV files contain monthly averages of the effective federal funds rate published by the Board of Governors of the Federal Reserve System in the H.15 Selected Interest Rates release. The series identifier is `H15/H15/RIFSPFF_N.M`; values are percentages per year.

## Development snapshot

`effr_monthly.csv` covers July 1954 through February 2017. This is the dataset used to develop the forecasting benchmark and choose its feature set, hyperparameters, and model-selection procedure.

- Retrieved: September 10, 2026
- [Official series preview](https://www.federalreserve.gov/datadownload/Preview.aspx?pi=400&preview=H15%2FH15%2FRIFSPFF_N.M&rel=H15)
- [Data Download Program](https://www.federalreserve.gov/datadownload/Download.aspx?rel=H15)

## External validation snapshot

`effr_external.csv` contains the next 114 observations, from March 2017 through August 2026. None of these observations was used to choose features, models, or hyperparameters. January and February 2017 were included in the source request as a boundary check; their published values, 0.65 and 0.66, match the final two rows of the development snapshot.

- Retrieved: September 18, 2026
- [H.15 monthly data package](https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15&series=d7e27b7b09a3a7feae95b9c61781fcd8&lastobs=&from=01%2F01%2F2017&to=09%2F01%2F2026&filetype=csv&label=include&layout=seriescolumn&type=package)
- Raw package SHA-256: `30abc7bece31600a5b019c9f00e114d71d41c8bb6587952fc88ab38dbdf48905`
- Cleaned file SHA-256: `3f7bd5aab67d3952b7ae814d795f74adb04c6ecfa88cf8fc77206a9531e6d257`

Both snapshots use the first calendar day as the timestamp for each source month and retain the published value to two decimal places. The external file was made by selecting the monthly EFFR column from the official package, renaming the date and value columns, dropping the two boundary-check rows, and writing the remaining values in chronological order.

The Board states that information on its website is in the public domain unless otherwise indicated and asks users to cite the Board as the source. See the [Federal Reserve Board disclaimer](https://www.federalreserve.gov/disclaimer.htm). The repository's MIT license applies to the code; the data files retain their source attribution and are provided under the source's terms.
