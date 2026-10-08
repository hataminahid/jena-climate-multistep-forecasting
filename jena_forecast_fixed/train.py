
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from seed_utils import set_seed

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
from data_utils import get_processed_data
from windowing import WindowGenerator
from models import build_naive_baseline, build_lstm, build_gru, Seq2Seq, compile_and_fit
from garch_module import (
    compute_residual_series,
    fit_garch,
    forecast_volatility,
    build_ghw_sample_weights,
    build_confidence_intervals,
)
from metrics_utils import compute_full_metrics, format_metrics, flatten_for_csv

set_seed()  

TARGET_COLUMNS = ["T (degC)", "rh (%)"] 
INPUT_WIDTH = 24       
HORIZONS = [6, 24, 72]  
BATCH_SIZE = 256
GHW_ALPHA = 1.0      

def evaluate(model, window, is_seq2seq=False):
    all_true, all_pred = [], []
    for x, y in window.test:
        pred = model(x, training=False)
        all_true.append(y.numpy())
        all_pred.append(pred.numpy())
    y_true = np.concatenate(all_true, axis=0)  # (N, label_width, num_label_features)
    y_pred = np.concatenate(all_pred, axis=0)

    mae_per_step = np.mean(np.abs(y_true - y_pred), axis=(0, 2))
    rmse_per_step = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 2)))
    return mae_per_step, rmse_per_step, y_true, y_pred


def fit_quick_one_step_model_and_get_residuals(train_df, val_df, test_df, num_features):

    one_step_window = WindowGenerator(
        input_width=INPUT_WIDTH, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=[TARGET_COLUMNS[0]], batch_size=BATCH_SIZE,
    )
    quick_lstm = build_lstm(INPUT_WIDTH, 1, num_features, 1, units=16)
    compile_and_fit(quick_lstm, one_step_window, max_epochs=8, patience=2)

    train_arr = train_df.values.astype(np.float32)
    target_idx = train_df.columns.get_loc(TARGET_COLUMNS[0])
    n = len(train_arr)
    xs = np.stack([train_arr[i:i + INPUT_WIDTH] for i in range(n - INPUT_WIDTH)])
    y_true_1step = train_arr[INPUT_WIDTH:, target_idx]
    y_pred_1step = quick_lstm.predict(xs, verbose=0)[:, 0, 0]
    return compute_residual_series(y_true_1step, y_pred_1step)


def run_for_horizon(train_df, val_df, test_df, label_width):
    print(f"\n===== افق = {label_width} گام =====")
    window = WindowGenerator(
        input_width=INPUT_WIDTH,
        label_width=label_width,
        shift=label_width,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=TARGET_COLUMNS,
        batch_size=BATCH_SIZE,
    )
    num_features = train_df.shape[1]
    num_label_features = len(TARGET_COLUMNS)

    results = {}

    # --- baseline ---
    baseline = build_naive_baseline(label_width, num_label_features)
    baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])
    mae, rmse, y_true, y_pred = evaluate(baseline, window)
    results["baseline"] = (mae, rmse, y_true, y_pred)
    print(f"baseline   MAE(avg)={mae.mean():.4f}  RMSE(avg)={rmse.mean():.4f}")

    # --- LSTM ---
    lstm = build_lstm(INPUT_WIDTH, label_width, num_features, num_label_features)
    compile_and_fit(lstm, window)
    mae, rmse, y_true, y_pred = evaluate(lstm, window)
    results["lstm"] = (mae, rmse, y_true, y_pred)
    print(f"LSTM       MAE(avg)={mae.mean():.4f}  RMSE(avg)={rmse.mean():.4f}")

    # --- GRU ---
    gru = build_gru(INPUT_WIDTH, label_width, num_features, num_label_features)
    compile_and_fit(gru, window)
    mae, rmse, y_true, y_pred = evaluate(gru, window)
    results["gru"] = (mae, rmse, y_true, y_pred)
    print(f"GRU        MAE(avg)={mae.mean():.4f}  RMSE(avg)={rmse.mean():.4f}")

    # --- Seq2Seq ---
    seq2seq = Seq2Seq(num_label_features=num_label_features, label_width=label_width)
    compile_and_fit(seq2seq, window)
    mae, rmse, y_true, y_pred = evaluate(seq2seq, window)
    results["seq2seq"] = (mae, rmse, y_true, y_pred)
    print(f"Seq2Seq    MAE(avg)={mae.mean():.4f}  RMSE(avg)={rmse.mean():.4f}")

    residuals = fit_quick_one_step_model_and_get_residuals(train_df, val_df, test_df, num_features)
    garch_res, sigma_t = fit_garch(residuals)

    sample_weights = build_ghw_sample_weights(
        sigma_t, input_width=INPUT_WIDTH, label_width=label_width, shift=label_width,
        alpha=GHW_ALPHA, component="combined",
    )

    train_x, train_y = [], []
    for x, y in window.make_dataset(train_df, shuffle=False):
        train_x.append(x.numpy())
        train_y.append(y.numpy())
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
   
    n_common = min(len(train_x), len(sample_weights))
    train_x, train_y, sample_weights = train_x[:n_common], train_y[:n_common], sample_weights[:n_common]

    lstm_ghw = build_lstm(INPUT_WIDTH, label_width, num_features, num_label_features)
    lstm_ghw.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, mode="min", restore_best_weights=True
    )
    
    lstm_ghw.fit(
        train_x, train_y, sample_weight=sample_weights,
        validation_data=window.val, batch_size=BATCH_SIZE,
        epochs=20, callbacks=[early_stopping], verbose=0,
    )

    mae_g, rmse_g, y_true_test, y_pred_test = evaluate(lstm_ghw, window)
    results["lstm_ghw"] = (mae_g, rmse_g, y_true_test, y_pred_test)
    print(f"LSTM+GHW   MAE(avg)={mae_g.mean():.4f}  RMSE(avg)={rmse_g.mean():.4f}  "
          f"(بهبود نسبت به LSTM ساده: {(results['lstm'][0].mean() - mae_g.mean()):+.4f} MAE)")

    sigma_h = forecast_volatility(garch_res, horizon=label_width)
    sample_point_forecast = y_pred_test[0]  # (label_width, num_label_features)
    lower, upper = build_confidence_intervals(sample_point_forecast, sigma_h)
    print(f"نمونه بازه‌ی اطمینان ۹۵٪ در اولین گام افق: "
          f"[{lower[0,0]:.3f}, {upper[0,0]:.3f}]  (پیش‌بینی نقطه‌ای={sample_point_forecast[0,0]:.3f})")

    return results


def plot_error_accumulation(all_results, horizon):
    plt.figure(figsize=(8, 5))
    for name, (mae, rmse, *_rest) in all_results.items():
        plt.plot(range(1, len(mae) + 1), mae, marker="o", label=name)
    plt.xlabel("گام افق")
    plt.ylabel("MAE")
    plt.title(f"انباشت خطا با افزایش افق (label_width={horizon})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"error_accumulation_h{horizon}.png"), dpi=150)
    plt.close()


def main():
    train_df, val_df, test_df, mean, std = get_processed_data()
    mean_vals = mean[TARGET_COLUMNS].values
    std_vals = std[TARGET_COLUMNS].values

    for h in HORIZONS:
        results = run_for_horizon(train_df, val_df, test_df, h)
        plot_error_accumulation(results, h)

        print(f"\n----- معیارهای تکمیلی (R²، SMAPE، دقت٪) در واحد فیزیکی -- افق={h} -----")
        rows = []
        for name, (mae, rmse, y_true, y_pred) in results.items():
            ext = compute_full_metrics(y_true, y_pred, mean_vals, std_vals, TARGET_COLUMNS)
            print(f"  {name:12s} {format_metrics(ext, TARGET_COLUMNS)}")
            row = {"model": name, "mae_avg": mae.mean(), "rmse_avg": rmse.mean()}
            row.update(flatten_for_csv(ext, TARGET_COLUMNS))
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, f"metrics_h{h}.csv"), index=False)


if __name__ == "__main__":
    main()
