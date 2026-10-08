Reproducing Results (Reproduce.md)
A complete instruction from zero to the final result. Everything is fixed with seed_utils.py (seed=42) — results should be identical across different runs on the same machine (minor differences on different GPUs are natural due to the cuDNN implementation).

2. Running the main architectures + the original contribution (GHW) on all 3 horizons
bash
python train.py
This script, for each horizon [6, 24, 72]: trains baseline/LSTM/GRU/Seq2Seq, then trains an LSTM with the GARCH-Aware Horizon-Weighted (GHW) loss — this project’s original contribution — and compares it with the rest.

Outputs: metrics_h6.csv, metrics_h24.csv, metrics_h72.csv, error_accumulation_h*.png — estimated time on a typical CPU: 15-30 minutes (dataset download ~13 MB + training 5 models × 3 horizons).

3. Final and unified script (recommended as the backbone of the report)
bash
python train_final.py
Runs all 4 architectures + the three-way GHW ablation (horizon-only / volatility-only / combined) + the confidence interval + the “where it fails” analysis on all three horizons [6, 24, 72] at once. It is slower than train.py because it trains three extra models (the ablation) instead of one. Final output: final_summary_all_horizons.csv (the main table of your report) + final_error_accumulation_h*.png.

4. Suggested order for the report
If time is limited, train_final.py alone is enough to cover the main elements of the problem statement (4 architectures, 3 horizons, one original innovation with a proven improvement + ablation, confidence interval, failure analysis). train.py is the lighter and faster version (no ablation) for quickly seeing the main result.

Troubleshooting notes
If arch shows a “convergence” warning during fit but no error, it is not a problem — it is due to the small scale of the normalized residuals and does not ruin the final prediction (it is already handled in the code by multiplying by 100).
The dataset is downloaded via tf.keras.utils.get_file; if the organizational network blocks access to storage.googleapis.com, manually get the CSV file from https://www.bgc-jena.mpg.de/wetter/ and pass its path directly to load_dataframe() in data_utils.py.
If you want to change the horizon weighting strength, change GHW_ALPHA at the top of train.py or train_final.py (default 1.0; 0 means volatility-only weighting, without horizon weight).
5. Supplementary metrics (R², SMAPE, accuracy%)
Both scripts (train.py, train_final.py) use metrics_utils.py and, in addition to MAE/RMSE (which are on z-score normalized data, suitable only for comparing models with each other), also print and save these metrics in the real physical unit (degrees Celsius for temperature, percent for humidity) in the same CSV file:

R² (coefficient of determination): how much of the real variance of the data is explained; closer to 1 is better.
SMAPE (symmetric percentage error): unlike ordinary MAPE, it does not blow up when the temperature is near zero degrees.
accuracy% (accuracy_pct = 100 − SMAPE): for more intuitive reporting.
The new CSV columns are added with the pattern {T (degC)|rh (%)|overall}_{mae|rmse|r2|smape|accuracy_pct}; no extra run or manual editing is needed — just run the scripts according to the steps above.