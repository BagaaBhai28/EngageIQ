# Sources and attribution

- **Dehouche, N. (2018). Facebook Live Sellers in Thailand.** UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5R60S. https://archive.ics.uci.edu/dataset/488/facebook+live+sellers+in+thailand
- **Moro, S., Rita, P., & Vala, B. (2016). Facebook Metrics.** UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5QK55. https://archive.ics.uci.edu/dataset/368/facebook+metrics

Both source datasets are CC BY 4.0: https://creativecommons.org/licenses/by/4.0/. Raw CSVs are preserved; processed derivatives remove empty columns and invalid/duplicate rows and add analytics fields. Attribution must travel with redistributed datasets.

The Live Sellers repository card lists 7,051 instances, but the actual downloaded CSV contains 7,050 data rows. Its observed date range is July 15, 2012 to June 13, 2018. The application uses actual file counts and dates. Metrics contains 500 rows; target component checks found zero discrepancies. The raw file has some missing component/paid values, handled through train-only imputation for predictor inputs; target remains the provided total.

UCI's metadata does not define weekday coding. The GAMLSS maintainers' fbdata documentation identifies Sunday=1 through Saturday=7: https://gamlss-dev.r-universe.dev/gamlss.data/gamlss.data.pdf. Analytics weekdays independently derive from timestamps using Monday=0 through Sunday=6.

Chart.js 4.4.8 is bundled from the official npm package via jsDelivr. Its MIT license is included in the vendor directory.
