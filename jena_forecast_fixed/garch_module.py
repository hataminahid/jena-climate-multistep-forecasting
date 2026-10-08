
import numpy as np
from arch import arch_model


def compute_residual_series(y_true_1step: np.ndarray, y_pred_1step: np.ndarray) -> np.ndarray:
    return (y_true_1step - y_pred_1step).astype(np.float64)


def fit_garch(residual_series: np.ndarray, p: int = 1, q: int = 1):

    scale = 100.0
    am = arch_model(residual_series * scale, mean="Zero", vol="GARCH", p=p, q=q, dist="normal")
    res = am.fit(disp="off")
    sigma_in_sample = res.conditional_volatility / scale  # sigma_t هم‌طول residual_series
    return res, sigma_in_sample

# وزن دهی بیشتر به گام های دور تر 
def forecast_volatility(res, horizon: int) -> np.ndarray:
    scale = 100.0
    f = res.forecast(horizon=horizon, reindex=False)
    variance_h = f.variance.values[-1]  # shape (horizon,)
    sigma_h = np.sqrt(variance_h) / scale
    return sigma_h

def build_horizon_weights(label_width: int, alpha: float = 1.0, mode: str = "linear") -> np.ndarray:
    """
    mode="linear": w(h) = 1 + alpha * h/(H-1)
    mode="sqrt":   w(h) = 1 + alpha * sqrt(h/(H-1))  
    """
    if label_width <= 1:
        return np.ones(label_width, dtype=np.float32)
    h = np.arange(label_width, dtype=np.float64)
    frac = h / (label_width - 1)
    if mode == "linear":
        w = 1.0 + alpha * frac
    elif mode == "sqrt":
        w = 1.0 + alpha * np.sqrt(frac)
    else:
        raise ValueError(f"mode ناشناخته: {mode}")
    return w.astype(np.float32)


def build_volatility_weights_per_step(sigma_t: np.ndarray, input_width: int,
                                       label_width: int, shift: int,
                                       eps: float = 1e-6) -> np.ndarray:
    total_window = input_width + shift
    n_windows = len(sigma_t) - total_window + 1
    label_start = total_window - label_width
    weights = np.empty((n_windows, label_width), dtype=np.float32)
    for i in range(n_windows):
        seg = sigma_t[i + label_start: i + total_window]
        weights[i] = 1.0 / (seg ** 2 + eps)
    return weights


def build_ghw_sample_weights(sigma_t: np.ndarray, input_width: int, label_width: int,
                              shift: int, alpha: float = 1.0, horizon_mode: str = "linear",
                              eps: float = 1e-6, component: str = "combined") -> np.ndarray:

    vol_w = build_volatility_weights_per_step(sigma_t, input_width, label_width, shift, eps)
    hor_w = build_horizon_weights(label_width, alpha=alpha, mode=horizon_mode)

    if component == "combined":
        combined = vol_w * hor_w[None, :]
    elif component == "horizon_only":
        combined = np.tile(hor_w[None, :], (vol_w.shape[0], 1))
    elif component == "volatility_only":
        combined = vol_w
    else:
        raise ValueError(f"component ناشناخته: {component}")

    combined = combined / (combined.mean() + eps)
    return combined.astype(np.float32)


def build_confidence_intervals(point_forecast: np.ndarray, sigma_h: np.ndarray,
                                z: float = 1.96) -> tuple:
    sigma_h = np.asarray(sigma_h)
    if point_forecast.ndim == 2:
        sigma_h = sigma_h[:, None]
    lower = point_forecast - z * sigma_h
    upper = point_forecast + z * sigma_h
    return lower, upper
