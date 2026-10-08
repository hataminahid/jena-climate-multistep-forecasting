
import numpy as np


def denormalize(arr, mean_vals, std_vals):
    return arr * std_vals + mean_vals


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    return float(np.mean(num / den) * 100.0)


def compute_full_metrics(y_true_norm, y_pred_norm, mean_vals, std_vals, feature_names):

    y_true = denormalize(y_true_norm, mean_vals, std_vals)
    y_pred = denormalize(y_pred_norm, mean_vals, std_vals)

    results = {}
    for i, name in enumerate(feature_names):
        yt, yp = y_true[..., i], y_pred[..., i]
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        r2 = float(r2_score(yt, yp))
        sm = smape(yt, yp)
        acc = max(0.0, 100.0 - sm)
        results[name] = {"mae": mae, "rmse": rmse, "r2": r2, "smape": sm, "accuracy_pct": acc}

    mae_o = float(np.mean([results[n]["mae"] for n in feature_names]))
    rmse_o = float(np.mean([results[n]["rmse"] for n in feature_names]))
    r2_o = float(np.mean([results[n]["r2"] for n in feature_names]))
    sm_o = float(np.mean([results[n]["smape"] for n in feature_names]))
    acc_o = max(0.0, 100.0 - sm_o)
    results["overall"] = {"mae": mae_o, "rmse": rmse_o, "r2": r2_o, "smape": sm_o, "accuracy_pct": acc_o}
    return results


def format_metrics(results, feature_names):
    parts = []
    for name in list(feature_names) + ["overall"]:
        m = results[name]
        parts.append(
            f"{name}: MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
            f"R2={m['r2']:.3f}  SMAPE={m['smape']:.2f}%  دقت={m['accuracy_pct']:.2f}%"
        )
    return "   |   ".join(parts)


def flatten_for_csv(results, feature_names, prefix=""):
    row = {}
    for name in list(feature_names) + ["overall"]:
        for metric_name, value in results[name].items():
            row[f"{prefix}{name}_{metric_name}"] = value
    return row
