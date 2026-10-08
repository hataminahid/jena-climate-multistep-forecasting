Multivariate Weather Forecasting with RNN + Original GARCH-Aware Horizon-Weighted Loss Contribution
For the full technical report (problem/data/method/results/limitations/references) see REPORT.md. For step-by-step reproduction instructions see REPRODUCE.md.

Project structure
text
seed_utils.py       fixing the random seed for full reproducibility
data_utils.py        downloading Jena Climate + cleaning + feature engineering + split/normalization
windowing.py          WindowGenerator for building input/label windows
models.py             baseline (naive persistence), LSTM, GRU, Seq2Seq (GRU encoder-decoder)
garch_module.py        original contribution: GARCH-Aware Horizon-Weighted Loss (GHW) + confidence interval
train.py               main script: baseline/LSTM/GRU/Seq2Seq + LSTM+GHW over horizons [6, 24, 72] steps
train_final.py          complete script: the same + three-way GHW ablation + confidence interval + model failure analysis
Run: pip install tensorflow arch pandas matplotlib then python train.py (downloading the dataset requires internet access.)

Hypothesis (before the experiment, per the instructions)
Multi-step weather forecast error has two kinds of non-uniform structure:

Horizon structure: as we move away from the forecast moment, uncertainty accumulates; farther horizon steps are both harder and of greater practical importance (e.g. a 3-day forecast), so they deserve more gradient weight — which plain MSE does not give them.
Temporal volatility structure: during stable atmospheric periods error variance is low, while near weather fronts/sudden pressure changes variance is high (volatility clustering, the phenomenon GARCH was built precisely to model).
Main hypothesis: if we fit a GARCH(1,1) on the one-step residual of a lightweight LSTM and combine its conditional variance with an explicit, increasing horizon weight (instead of using either one alone), the resulting loss (“GARCH-Aware Horizon-Weighted” / GHW) weights both the farther horizon steps and the intrinsically unpredictable (high-volatility) periods in a more targeted way, and the model gives lower MAE/RMSE than the plain LSTM baseline (no weighting) and than each of the two mechanisms alone (horizon-only weighting, or volatility-only weighting) — especially at longer horizons (24, 72 steps) and in high-GARCH-volatility time intervals.

The comparison baseline: a plain LSTM (unweighted MSE) with an exactly identical architecture. train_final.py also runs a three-way ablation (horizon-only / volatility-only / combined-GHW) to establish that the benefit really comes from combining the two mechanisms, not just one of them.

Answers to the discussion questions
Attention — what would “looking back” be expected to attend to? An attention mechanism over past steps would likely learn to weight more heavily the steps that are phase-aligned with the daily/seasonal cycle (e.g. the same hour of the previous day, or the peak/trough of the previous day’s temperature) and also the moments of rapid slope change in air pressure/humidity, which usually signal an incoming weather front. Unlike recurrence, which accumulates past information in a compressed, exponentially-forgetting hidden state, attention can point directly and without signal decay to a distant point (e.g. 24 hours ago). The improvement is expected to be larger at longer horizons, because at short horizons the last few observed steps already carry almost all the needed information (high local autocorrelation) and plain recurrence captures it well; but at long horizons, dependence on the farther cyclic pattern (not just the previous step) becomes more important, and it is there that direct, decay-free access to distant steps shows its advantage.

The sequential inductive bias of RNN versus Transformer. An RNN, by processing step-by-step, carries an inductive bias of order and recurrence: it assumes the representation at time t should only be a function of the representation at t-1 and the new input. This means that on short sequences and limited data, the RNN generalizes with fewer samples, because its hypothesis space is more constrained and better pre-aligned with the Markov/local nature of physical time series (like temperature, whose changes are continuous and local). The Transformer has no such bias — it must learn sequence order and locality from scratch from the data (helped only by positional encoding), which requires much more data and longer sequences before its scalability advantage (parallel attention, long-range dependence without gradient decay) shows itself. Therefore: with limited data (like a few years of a single station) and relatively short sequences (tens to a few hundred steps), I still bet on the RNN; when large multi-station/multi-year data and very long sequences are involved (the multi-station generalization goal in the bonus part), the Transformer gains the edge.

Original contribution — GARCH-Aware Horizon-Weighted Loss (GHW)
File: garch_module.py (weighting logic) + train.py/train_final.py (training/evaluation)

Pipeline steps:

A plain LSTM with a normal loss (unweighted MSE) is trained on each horizon as the baseline (lstm in the section 1 results).
A lightweight one-step LSTM is trained on the train data to extract continuous residuals; a GARCH(1,1) is fit on these residuals and the conditional variance series σ_t is obtained (fit_garch).
For each horizon step, the explicit horizon weight w_horizon(h) = 1 + α·h/(H-1) is computed (build_horizon_weights) — the last horizon step is weighted (1+α) times the first step.
The final weight of each (window, horizon step) is the product of the horizon weight and the inverse conditional-variance weight of GARCH at that same time instant: w(i,h) = w_horizon(h) / σ_t(i,h)² (build_ghw_sample_weights, component="combined"), then normalized so the total mean weight stays 1 (the effective learning rate does not change).
This weight matrix is passed directly as sample_weight (shape (N, label_width)) to model.fit — because Keras, for MeanSquaredError on a 3D output (batch, label_width, features), first reduces the feature axis and the remaining shape (batch, label_width) exactly matches our weight shape; so without a manual training loop (GradientTape), each horizon step gets its own independent weight in the gradient.
LSTM+GHW is trained with the same architecture, epoch count and early-stopping exactly like the baseline, and compared in MAE/RMSE (overall and per horizon step).
Ablation (train_final.py, GHW_COMPONENTS): three models are compared side by side — LSTM+horizon-only (component="horizon_only", ignoring σ_t) / LSTM+volatility-only (component="volatility_only", ignoring the horizon weight) / LSTM+combined-GHW (component="combined") — to establish how much the combination of the two mechanisms benefits beyond each one alone.

Limitation note (for the report): the GARCH parameters are fit over the whole train range at once (not rolling/expanding-window), so the σ_t values at the start of train are slightly optimistic; a production version should refit with an expanding window. Also α (the horizon weighting strength, default 1.0) is a hyperparameter — for the report you can also try several α values and report the sensitivity of the result (optional bonus goal).

Final and complete script — train_final.py
This file brings together all the elements of the problem statement in one run:

baseline + LSTM + GRU + Seq2Seq over all three increasing horizons [6, 24, 72]
on the best architecture (BEST_ARCH, default "lstm"), the three-way GHW ablation (horizon-only / volatility-only / combined) is trained
95% confidence interval with GARCH on the actual prediction of the combined-GHW model (bonus goal)
automatic “where the model fails” analysis: the ratio of absolute error in the top 10% highest-volatility periods (per GARCH) to the rest of the data — a proxy for sudden fronts
final summary table in final_summary_all_horizons.csv (including error growth from the first to the last horizon step, per model and per horizon) + plot final_error_accumulation_h*.png
If after running you see that GRU or Seq2Seq beat LSTM in the baseline part, change the BEST_ARCH value at the top of the file and rerun so GHW is applied to the actual best architecture, not a default assumption.

Report notes (fill in after running)
Window choice: input_width=24 steps (24 hours with hourly subsampling), horizons [6, 24, 72] steps meaning 6h/1day/3days.
Baseline table vs LSTM/GRU/Seq2Seq/LSTM+GHW (and the ablation in train_final.py) in metrics_h*.csv / final_summary_all_horizons.csv.
Error accumulation plot in error_accumulation_h*.png — check how steeply MAE grows as the horizon step increases and compare the models (expectation: GHW should have a gentler slope than plain LSTM at the last horizon steps, since that is exactly where it received more weight during training).
For “where the model fails”: compare the time intervals where σ_t from GARCH came out high (sigma_t in garch_module.py) with the model’s actual error — these are candidates for sudden fronts.
