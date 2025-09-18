# Bucketed Performance by Pair and Regime

This report segments performance by 2-year time buckets, market conditions (trend vs ranging), and volatility regimes (low/medium/high).

Notes:
- Returns are computed from equity first/last within each bucket. Trades are estimated from signal-turn counts within the bucket.
- Market conditions and volatility regimes are computed from 20-day rolling trend and volatility as per the earlier exploratory script.
- Definitions: Trending Up (> +2% over 20d), Trending Down (< -2% over 20d), else Ranging. Low vol (<5% annualized), High vol (>10%), else Medium.

## Cross-Pair Summaries

### Cross-Pair Summary — 2-Year Time Buckets

| Bucket | Median Return % | Mean Return % | Total Trades | Total Days | Pair Coverage |
|--------|-----------------:|--------------:|------------:|----------:|--------------:|
| 2015-2016 | 5.10 | 4.05 | 58 | 3513 | 7 |
| 2017-2018 | -4.15 | -2.14 | 97 | 3499 | 7 |
| 2019-2020 | -0.30 | 2.20 | 58 | 3528 | 7 |
| 2021-2022 | -0.30 | 1.09 | 72 | 3521 | 7 |
| 2023-2024 | -0.11 | 1.37 | 107 | 3516 | 7 |
| 2025 | 1.79 | 2.25 | 14 | 1084 | 7 |

### Cross-Pair Summary — Market Condition Buckets

| Bucket | Median Return % | Mean Return % | Total Trades | Total Days | Pair Coverage |
|--------|-----------------:|--------------:|------------:|----------:|--------------:|
| Ranging | 6.51 | 8.90 | 179 | 8326 | 7 |
| Trending Down | 4.91 | 8.45 | 99 | 4191 | 7 |
| Trending Up | 4.30 | 7.98 | 155 | 6004 | 7 |
| Unknown | 0.00 | 0.00 | 0 | 140 | 7 |

### Cross-Pair Summary — Volatility Regime Buckets

| Bucket | Median Return % | Mean Return % | Total Trades | Total Days | Pair Coverage |
|--------|-----------------:|--------------:|------------:|----------:|--------------:|
| High Vol | 6.51 | 8.68 | 385 | 17708 | 7 |
| Medium Vol | 0.62 | 0.22 | 24 | 813 | 6 |
| Unknown | 0.00 | 0.00 | 0 | 140 | 7 |

## Per-Pair Breakdowns

## audusd_gold

### 2-Year Time Buckets

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| 2015-2016 | 2.54 | 16 | 502 |
| 2017-2018 | -12.99 | 16 | 499 |
| 2019-2020 | 25.55 | 11 | 504 |
| 2021-2022 | -8.36 | 8 | 503 |
| 2023-2024 | 2.20 | 9 | 502 |
| 2025 | 1.53 | 3 | 155 |

### Market Condition Buckets (Trend/Ranging)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| Ranging | 6.51 | 24 | 1451 |
| Trending Down | 4.91 | 18 | 496 |
| Trending Up | 4.30 | 25 | 698 |
| Unknown | 0.00 | 0 | 20 |

### Volatility Regime Buckets (Low/Med/High)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| High Vol | 6.51 | 43 | 1878 |
| Medium Vol | 2.14 | 21 | 767 |
| Unknown | 0.00 | 0 | 20 |

## usdbrl_soybeans

### 2-Year Time Buckets

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| 2015-2016 | 5.10 | 12 | 502 |
| 2017-2018 | -5.77 | 14 | 500 |
| 2019-2020 | -3.39 | 9 | 504 |
| 2021-2022 | -2.75 | 16 | 503 |
| 2023-2024 | -2.35 | 31 | 502 |
| 2025 | 3.54 | 2 | 154 |

### Market Condition Buckets (Trend/Ranging)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| Ranging | -5.93 | 37 | 1371 |
| Trending Down | -5.93 | 23 | 517 |
| Trending Up | -9.14 | 27 | 757 |
| Unknown | 0.00 | 0 | 20 |

### Volatility Regime Buckets (Low/Med/High)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| High Vol | -5.93 | 83 | 2612 |
| Medium Vol | -8.74 | 1 | 33 |
| Unknown | 0.00 | 0 | 20 |

## usdcad_wti

### 2-Year Time Buckets

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| 2015-2016 | 0.62 | 2 | 502 |
| 2017-2018 | 3.22 | 15 | 500 |
| 2019-2020 | -10.53 | 4 | 504 |
| 2021-2022 | 1.33 | 9 | 503 |
| 2023-2024 | 5.09 | 8 | 502 |
| 2025 | -1.79 | 2 | 155 |

### Market Condition Buckets (Trend/Ranging)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| Ranging | -1.47 | 14 | 818 |
| Trending Down | -1.47 | 7 | 787 |
| Trending Up | 2.30 | 22 | 1041 |
| Unknown | 0.00 | 0 | 20 |

### Volatility Regime Buckets (Low/Med/High)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| High Vol | -1.47 | 41 | 2643 |
| Medium Vol | 1.23 | 1 | 3 |
| Unknown | 0.00 | 0 | 20 |

## usdclp_copper

### 2-Year Time Buckets

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| 2015-2016 | 9.99 | 20 | 502 |
| 2017-2018 | -0.37 | 13 | 499 |
| 2019-2020 | -0.30 | 2 | 504 |
| 2021-2022 | -0.91 | 7 | 503 |
| 2023-2024 | -0.14 | 22 | 503 |
| 2025 | 4.49 | 2 | 155 |

### Market Condition Buckets (Trend/Ranging)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| Ranging | 12.95 | 39 | 1609 |
| Trending Down | 12.95 | 16 | 358 |
| Trending Up | 12.95 | 18 | 679 |
| Unknown | 0.00 | 0 | 20 |

### Volatility Regime Buckets (Low/Med/High)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| High Vol | 12.95 | 66 | 2642 |
| Medium Vol | 0.00 | 0 | 4 |
| Unknown | 0.00 | 0 | 20 |

## usdmxn_wti

### 2-Year Time Buckets

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| 2015-2016 | -6.75 | 1 | 502 |
| 2017-2018 | -4.15 | 13 | 502 |
| 2019-2020 | 3.38 | 5 | 504 |
| 2021-2022 | -0.30 | 3 | 503 |
| 2023-2024 | -0.11 | 12 | 502 |
| 2025 | 6.06 | 1 | 155 |

### Market Condition Buckets (Trend/Ranging)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| Ranging | -2.41 | 10 | 826 |
| Trending Down | -2.41 | 9 | 785 |
| Trending Up | -2.41 | 17 | 1037 |
| Unknown | 0.00 | 0 | 20 |

### Volatility Regime Buckets (Low/Med/High)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| High Vol | -2.41 | 35 | 2648 |
| Unknown | 0.00 | 0 | 20 |

## usdnok_brent

### 2-Year Time Buckets

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| 2015-2016 | 6.81 | 3 | 502 |
| 2017-2018 | -5.64 | 20 | 500 |
| 2019-2020 | -0.31 | 11 | 504 |
| 2021-2022 | 11.98 | 13 | 503 |
| 2023-2024 | 7.36 | 7 | 503 |
| 2025 | 1.79 | 2 | 155 |

### Market Condition Buckets (Trend/Ranging)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| Ranging | 22.96 | 20 | 949 |
| Trending Down | 22.96 | 6 | 682 |
| Trending Up | 22.96 | 34 | 1016 |
| Unknown | 0.00 | 0 | 20 |

### Volatility Regime Buckets (Low/Med/High)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| High Vol | 22.96 | 55 | 2644 |
| Medium Vol | 6.69 | 1 | 3 |
| Unknown | 0.00 | 0 | 20 |

## usdzar_platinum

### 2-Year Time Buckets

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| 2015-2016 | 10.03 | 4 | 501 |
| 2017-2018 | 10.72 | 6 | 499 |
| 2019-2020 | 1.01 | 16 | 504 |
| 2021-2022 | 6.65 | 16 | 503 |
| 2023-2024 | -2.45 | 18 | 502 |
| 2025 | 0.11 | 2 | 155 |

### Market Condition Buckets (Trend/Ranging)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| Ranging | 29.71 | 35 | 1302 |
| Trending Down | 28.17 | 20 | 566 |
| Trending Up | 24.87 | 12 | 776 |
| Unknown | 0.00 | 0 | 20 |

### Volatility Regime Buckets (Low/Med/High)

| Bucket | Return % | Trades | Days |
|--------|----------:|-------:|-----:|
| High Vol | 28.17 | 62 | 2641 |
| Medium Vol | 0.00 | 0 | 3 |
| Unknown | 0.00 | 0 | 20 |


Generated at: 2025-09-18T00:24:14.081112Z