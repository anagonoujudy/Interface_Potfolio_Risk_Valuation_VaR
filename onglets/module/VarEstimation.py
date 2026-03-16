from __future__ import annotations
import streamlit as st

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import genpareto, jarque_bera, kurtosis, norm, skew, t


TRADING_DAYS = 252


# ============================================================
# STRUCTURES
# ============================================================

@dataclass
class VaRMethodResult:
    method: str
    category: str
    conditional: bool
    horizon: int
    status: str              # success / warning / error
    message: str
    var_return: float | None
    var_money: float | None
    var_series: pd.Series | None
    diagnostics: dict[str, Any]


# ============================================================
# OUTILS GÉNÉRAUX
# ============================================================

def _validate_weights(weights: dict[str, float]) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    if w.empty:
        raise ValueError("Le dictionnaire des poids est vide.")
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError("La somme des poids doit être égale à 1.")
    return w


def build_portfolio_series(
    asset_price_df: pd.DataFrame,
    weights: dict[str, float],
    capital_invested: float = 10000.0,
) -> dict[str, pd.Series]:
    """
    Construit :
    - rendements log des actifs
    - rendements log du portefeuille
    - valeur historique du portefeuille
    - pnl historique
    """
    w = _validate_weights(weights)

    missing_assets = [asset for asset in w.index if asset not in asset_price_df.columns]
    if missing_assets:
        raise ValueError(f"Actifs absents de la base de prix : {missing_assets}")

    prices = asset_price_df[w.index].dropna(how="any").copy()
    if prices.empty:
        raise ValueError("Aucune donnée de prix exploitable pour le portefeuille.")

    asset_log_returns = np.log(prices / prices.shift(1)).dropna()
    portfolio_log_returns = asset_log_returns @ w

    portfolio_value = capital_invested * np.exp(portfolio_log_returns.cumsum())
    portfolio_value = pd.concat(
        [pd.Series([capital_invested], index=[prices.index[0]]), portfolio_value]
    ).sort_index()

    portfolio_pnl = portfolio_value - capital_invested

    return {
        "asset_prices": prices,
        "asset_log_returns": asset_log_returns,
        "portfolio_returns": portfolio_log_returns,
        "portfolio_value": portfolio_value,
        "portfolio_pnl": portfolio_pnl,
    }


def _last_portfolio_value(portfolio_value: pd.Series) -> float:
    return float(portfolio_value.iloc[-1])


#def _return_var_to_money(var_return: float | None, portfolio_value: float, capital_invested:float) -> float | None:
def _return_var_to_money(var_return: float | None, capital_invested:float) -> float | None:

    if var_return is None or not np.isfinite(var_return):
        return None
    #return float(var_return * portfolio_value)
    return float(var_return * capital_invested)


def _portfolio_diagnostics(returns: pd.Series) -> dict[str, Any]:
    clean = returns.dropna()
    if clean.empty:
        return {}

    jb_stat, jb_p = jarque_bera(clean)
    return {
        "n_obs": int(len(clean)),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)),
        "skewness": float(skew(clean, bias=False)),
        "kurtosis_excess": float(kurtosis(clean, fisher=True, bias=False)),
        "jarque_bera_pvalue": float(jb_p),
    }


def compute_horizon_returns(returns: pd.Series, horizon: int) -> pd.Series:
    """
    Rendements log agrégés sur horizon glissant.
    """
    if horizon < 1:
        raise ValueError("L'horizon doit être >= 1.")
    if horizon == 1:
        return returns.dropna()
    return returns.rolling(horizon).sum().dropna()


def compute_forward_horizon_returns(returns: pd.Series, horizon: int) -> pd.Series:
    """
    Rendement futur sur h jours, aligné à la date t :
    r_{t,t+h} = r_t + ... + r_{t+h-1}
    """
    if horizon < 1:
        raise ValueError("L'horizon doit être >= 1.")
    if horizon == 1:
        return returns.dropna()

    fwd = pd.Series(index=returns.index, dtype=float)
    values = returns.values
    for i in range(len(values) - horizon + 1):
        fwd.iloc[i] = float(np.sum(values[i:i + horizon]))
    return fwd.dropna()


def _safe_window(series_len: int, target_window: int, min_window: int = 80) -> int:
    """
    Adapte la fenêtre à la taille effective de la série.
    """
    if series_len < min_window:
        raise ValueError(
            f"Historique insuffisant : {series_len} observations disponibles, minimum requis {min_window}."
        )
    return int(min(target_window, max(min_window, series_len // 2)))


def _make_error_result(
    method: str,
    category: str,
    conditional: bool,
    horizon: int,
    message: str,
    diagnostics: dict[str, Any] | None = None,
) -> VaRMethodResult:
    return VaRMethodResult(
        method=method,
        category=category,
        conditional=conditional,
        horizon=horizon,
        status="error",
        message=message,
        var_return=None,
        var_money=None,
        var_series=None,
        diagnostics=diagnostics or {},
    )


# ============================================================
# 1) VAR NON PARAMÉTRIQUE : HISTORIQUE
# ============================================================

def var_historical(
    returns: pd.Series,
    confidence: float,
    horizon: int,
    portfolio_value: float,
    window: int = 250,
) -> VaRMethodResult:
    alpha = 1.0 - confidence
    h_returns = compute_horizon_returns(returns, horizon)
    eff_window = _safe_window(len(h_returns), window)

    var_series = -h_returns.rolling(eff_window).quantile(alpha).shift(1).dropna()
    if var_series.empty:
        raise ValueError("Aucune prévision VaR historique disponible après rolling.")

    var_return = float(var_series.iloc[-1])

    diag = _portfolio_diagnostics(h_returns)
    diag.update({"window": eff_window, "alpha": alpha})

    return VaRMethodResult(
        method="Historique",
        category="Non paramétrique",
        conditional=False,
        horizon=horizon,
        status="success",
        message="VaR historique estimée avec succès.",
        var_return=var_return,
        var_money=_return_var_to_money(var_return, st.session_state["capital_invested"]),
        var_series=var_series,
        diagnostics=diag,
    )


# ============================================================
# 2) VAR SEMI-PARAMÉTRIQUE : VARIANCE-COVARIANCE
# ============================================================

def var_variance_covariance(
    returns: pd.Series,
    confidence: float,
    horizon: int,
    portfolio_value: float,
    window: int = 250,
) -> VaRMethodResult:
    alpha = 1.0 - confidence
    z_alpha = float(norm.ppf(alpha))

    eff_window = _safe_window(len(returns), window)

    mu = returns.rolling(eff_window).mean().shift(1)
    sigma = returns.rolling(eff_window).std(ddof=1).shift(1)

    var_series_1d = -(mu + sigma * z_alpha)
    var_series = (np.sqrt(horizon) * var_series_1d).dropna()

    if var_series.empty:
        raise ValueError("Aucune prévision VaR VCV disponible après rolling.")

    var_return = float(var_series.iloc[-1])

    diag = _portfolio_diagnostics(returns)
    diag.update({"window": eff_window, "z_alpha": z_alpha})

    return VaRMethodResult(
        method="Variance-Covariance",
        category="Semi-paramétrique",
        conditional=True,
        horizon=horizon,
        status="success",
        message="VaR variance-covariance estimée avec succès.",
        var_return=var_return,
        var_money=_return_var_to_money(var_return, st.session_state["capital_invested"]),
        var_series=var_series,
        diagnostics=diag,
    )


# ============================================================
# 3) VAR SEMI-PARAMÉTRIQUE : CORNISH-FISHER
# ============================================================

def _cornish_fisher_var_from_sample(sample: pd.Series, alpha: float) -> tuple[float, dict]:
    mu = float(sample.mean())
    sigma = float(sample.std(ddof=1))
    s = float(skew(sample, bias=False))
    k = float(kurtosis(sample, fisher=True, bias=False))
    z = float(norm.ppf(alpha))

    z_cf = (
        z
        + (z**2 - 1.0) * s / 6.0
        + (z**3 - 3.0 * z) * k / 24.0
        - (2.0 * z**3 - 5.0 * z) * (s**2) / 36.0
    )

    var_cf = -(mu + sigma * z_cf)

    return var_cf, {
        "mu": mu,
        "sigma": sigma,
        "skewness": s,
        "kurtosis_excess": k,
        "z_cf": z_cf,
    }


def var_cornish_fisher(
    returns: pd.Series,
    confidence: float,
    horizon: int,
    portfolio_value: float,
    window: int = 250,
) -> VaRMethodResult:
    alpha = 1.0 - confidence
    eff_window = _safe_window(len(returns), window)

    values = []
    idx = []

    for i in range(eff_window, len(returns)):
        sample = returns.iloc[i - eff_window:i]
        v, _ = _cornish_fisher_var_from_sample(sample, alpha)
        values.append(np.sqrt(horizon) * v)
        idx.append(returns.index[i])

    var_series = pd.Series(values, index=idx).dropna()
    if var_series.empty:
        raise ValueError("Aucune prévision VaR Cornish-Fisher disponible après rolling.")

    var_return = float(var_series.iloc[-1])

    _, last_diag = _cornish_fisher_var_from_sample(returns.iloc[-eff_window:], alpha)
    diag = _portfolio_diagnostics(returns)
    diag.update({"window": eff_window, **last_diag})

    return VaRMethodResult(
        method="Cornish-Fisher",
        category="Semi-paramétrique",
        conditional=True,
        horizon=horizon,
        status="success",
        message="VaR Cornish-Fisher estimée avec succès.",
        var_return=var_return,
        var_money=_return_var_to_money(var_return, st.session_state["capital_invested"]),
        var_series=var_series,
        diagnostics=diag,
    )


# ============================================================
# 4) VAR PARAMÉTRIQUE : RISKMETRICS / EWMA
# ============================================================

def var_riskmetrics(
    returns: pd.Series,
    confidence: float,
    horizon: int,
    portfolio_value: float,
    lambda_rm: float = 0.94,
) -> VaRMethodResult:
    alpha = 1.0 - confidence
    z_alpha = float(norm.ppf(alpha))

    r = returns.dropna()
    sigma2 = float(r.var(ddof=1))
    sigma2_path = []

    for rt in r:
        sigma2 = lambda_rm * sigma2 + (1.0 - lambda_rm) * (rt**2)
        sigma2_path.append(sigma2)

    sigma_1d = pd.Series(np.sqrt(sigma2_path), index=r.index).shift(1).dropna()
    var_series = (-(z_alpha * sigma_1d) * np.sqrt(horizon)).dropna()

    if var_series.empty:
        raise ValueError("Aucune prévision VaR RiskMetrics disponible.")

    var_return = float(var_series.iloc[-1])

    diag = _portfolio_diagnostics(r)
    diag.update({"lambda": lambda_rm, "z_alpha": z_alpha})

    return VaRMethodResult(
        method="RiskMetrics",
        category="Paramétrique",
        conditional=True,
        horizon=horizon,
        status="success",
        message="VaR RiskMetrics estimée avec succès.",
        var_return=var_return,
        var_money=_return_var_to_money(var_return, st.session_state["capital_invested"]),
        var_series=var_series,
        diagnostics=diag,
    )


# ============================================================
# OUTILS GARCH / EVT-GARCH
# ============================================================

def _detect_mean_model(returns: pd.Series, alpha: float = 0.05) -> tuple[str, int]:
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(returns.dropna(), lags=[10], return_df=True)
        if lb_test["lb_pvalue"].iloc[0] < alpha:
            return "AR", 1
    except Exception:
        pass
    return "Constant", 0


def _fit_garch_student(returns: pd.Series):
    mean_model, lags = _detect_mean_model(returns)
    scaled = returns.dropna() * 100.0

    model = arch_model(
        scaled,
        mean="AR" if mean_model == "AR" else "Constant",
        lags=lags,
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
    )
    fit = model.fit(disp="off")

    cond_vol = fit.conditional_volatility / 100.0
    resid = fit.resid / 100.0
    nu = float(fit.params.get("nu"))
    mu_const = fit.params.get("Const", fit.params.get("mu", 0.0)) / 100.0

    # paramètres GARCH utiles pour agrégation horizon
    params = fit.params.to_dict()
    omega = float(params.get("omega", np.nan)) / (100.0**2)
    alpha_g = float(params.get("alpha[1]", np.nan))
    beta_g = float(params.get("beta[1]", np.nan))

    return fit, mean_model, lags, cond_vol, resid, nu, mu_const, omega, alpha_g, beta_g


def _aggregate_garch_variance(
    sigma1_sq: pd.Series,
    omega: float,
    alpha_g: float,
    beta_g: float,
    horizon: int,
) -> pd.Series:
    """
    Agrégation temporelle GARCH(1,1) :
    h_{t+h|t} = sigma_bar^2 * [1 - (a+b)^(h-1)] + (a+b)^(h-1) * h_{t+1|t}
    puis somme des variances conditionnelles de t+1 à t+h.
    Référence : structure par terme GARCH du cours. 
    """
    if horizon == 1:
        return sigma1_sq

    if not np.isfinite(omega) or not np.isfinite(alpha_g) or not np.isfinite(beta_g):
        # fallback prudent
        return horizon * sigma1_sq

    ab = alpha_g + beta_g
    if ab >= 0.999:
        return horizon * sigma1_sq

    sigma_bar_sq = omega / (1.0 - ab)

    total = pd.Series(0.0, index=sigma1_sq.index)
    for k in range(1, horizon + 1):
        hk = sigma_bar_sq * (1.0 - (ab ** (k - 1))) + (ab ** (k - 1)) * sigma1_sq
        total = total + hk

    return total


def _evt_pot_fit_on_losses(
    losses: pd.Series,
    threshold_quantile: float = 0.95,
    min_excess: int | None = None,
) -> dict:
    """
    Ajustement GPD sur les pertes excédant un seuil.
    min_excess est rendu adaptatif pour éviter les échecs artificiels.
    """
    losses = losses.dropna().astype(float)
    if losses.empty:
        raise ValueError("Série de pertes vide pour EVT.")

    u = float(losses.quantile(threshold_quantile))
    exceedances = losses[losses > u] - u
    nu = len(exceedances)
    n = len(losses)

    expected_excess = max(1, int(np.floor((1.0 - threshold_quantile) * n)))
    if min_excess is None:
        min_excess_eff = max(8, min(expected_excess, 20))
    else:
        min_excess_eff = min(min_excess, max(8, expected_excess))

    if nu < min_excess_eff:
        raise ValueError(
            f"Nombre de dépassements insuffisant pour EVT ({nu} < {min_excess_eff}). "
            f"Seuil={threshold_quantile:.2f}, taille fenêtre={n}."
        )

    xi, loc, beta = genpareto.fit(exceedances, floc=0.0)

    return {
        "threshold_u": u,
        "threshold_quantile": threshold_quantile,
        "n_obs": n,
        "n_excess": nu,
        "min_excess_effective": min_excess_eff,
        "xi": float(xi),
        "beta": float(beta),
    }


def _evt_pot_quantile_from_fit(confidence: float, fit_info: dict) -> float:
    alpha = 1.0 - confidence
    u = fit_info["threshold_u"]
    n = fit_info["n_obs"]
    nu = fit_info["n_excess"]
    xi = fit_info["xi"]
    beta = fit_info["beta"]

    if abs(xi) > 1e-8:
        q_loss = u + (beta / xi) * (((nu / n) / alpha) ** xi - 1.0)
    else:
        q_loss = u + beta * np.log((nu / n) / alpha)

    return float(q_loss)


# ============================================================
# 5) VAR PARAMÉTRIQUE : GARCH-STUDENT DYNAMIQUE
# ============================================================

def var_garch_student(
    returns: pd.Series,
    confidence: float,
    horizon: int,
    portfolio_value: float,
) -> VaRMethodResult:
    alpha = 1.0 - confidence
    fit, mean_model, lags, cond_vol, resid, nu, mu_const, omega, alpha_g, beta_g = _fit_garch_student(returns)

    q_t = float(np.sqrt((nu - 2.0) / nu) * t.ppf(alpha, df=nu))

    sigma1_sq = (cond_vol.shift(1).dropna() ** 2)
    sigma_h_sq = _aggregate_garch_variance(sigma1_sq, omega, alpha_g, beta_g, horizon)
    sigma_h = np.sqrt(sigma_h_sq)

    mu_h = horizon * mu_const
    var_series = -(mu_h + sigma_h * q_t)
    var_series = var_series.dropna()

    if var_series.empty:
        raise ValueError("Aucune prévision VaR GARCH disponible.")

    var_return = float(var_series.iloc[-1])

    diag = _portfolio_diagnostics(returns)
    diag.update(
        {
            "mean_model": mean_model,
            "lags": lags,
            "nu": nu,
            "omega": omega,
            "alpha_garch": alpha_g,
            "beta_garch": beta_g,
            "aic": float(fit.aic),
            "bic": float(fit.bic),
        }
    )

    return VaRMethodResult(
        method="GARCH-Student",
        category="Paramétrique",
        conditional=True,
        horizon=horizon,
        status="success",
        message="VaR GARCH-Student estimée avec succès.",
        var_return=var_return,
        var_money=_return_var_to_money(var_return, st.session_state["capital_invested"]),
        var_series=var_series,
        diagnostics=diag,
    )


# ============================================================
# 6) VAR SEMI-PARAMÉTRIQUE : EVT DYNAMIQUE (ROLLING POT)
# ============================================================

def var_evt(
    returns: pd.Series,
    confidence: float,
    horizon: int,
    portfolio_value: float,
    window: int = 250,
    threshold_quantile: float = 0.95,
    min_excess: int | None = None,
) -> VaRMethodResult:
    h_returns = compute_horizon_returns(returns, horizon)
    losses = -h_returns
    eff_window = _safe_window(len(losses), window)

    values = []
    idx = []
    last_fit = None
    failures = 0

    for i in range(eff_window, len(losses)):
        sample_losses = losses.iloc[i - eff_window:i]
        try:
            fit_info = _evt_pot_fit_on_losses(
                sample_losses,
                threshold_quantile=threshold_quantile,
                min_excess=min_excess,
            )
            q_loss = _evt_pot_quantile_from_fit(confidence, fit_info)
            values.append(q_loss)
            idx.append(losses.index[i])
            last_fit = fit_info
        except Exception:
            failures += 1
            continue

    if not values:
        raise ValueError(
            "Impossible d'estimer la VaR EVT sur la fenêtre choisie. "
            "Le seuil est probablement trop élevé ou l'échantillon utile est trop court."
        )

    var_series = pd.Series(values, index=idx).dropna()
    var_return = float(var_series.iloc[-1])

    diag = _portfolio_diagnostics(h_returns)
    if last_fit is not None:
        diag.update(last_fit)
    diag.update({"window": eff_window, "n_failed_windows": failures})

    status = "warning" if failures > 0 else "success"
    message = (
        "VaR EVT estimée avec succès."
        if failures == 0
        else "VaR EVT estimée avec succès, avec quelques fenêtres non exploitables ignorées."
    )

    return VaRMethodResult(
        method="EVT-POT",
        category="Semi-paramétrique",
        conditional=True,
        horizon=horizon,
        status=status,
        message=message,
        var_return=var_return,
        var_money=_return_var_to_money(var_return, st.session_state["capital_invested"]),
        var_series=var_series,
        diagnostics=diag,
    )


# ============================================================
# 7) VAR PARAMÉTRIQUE : EVT-GARCH DYNAMIQUE
# ============================================================

def var_evt_garch(
    returns: pd.Series,
    confidence: float,
    horizon: int,
    portfolio_value: float,
    window: int = 250,
    threshold_quantile: float = 0.95,
    min_excess: int | None = None,
) -> VaRMethodResult:
    """
    EVT-GARCH dynamique :
    - ajuste un GARCH(1,1)-Student
    - standardise les résidus
    - applique EVT-POT sur les pertes des résidus standardisés en rolling
    - reconstruit une VaR dynamique sur l’horizon choisi
    """
    fit, mean_model, lags, cond_vol, resid, nu, mu_const, omega, alpha_g, beta_g = _fit_garch_student(returns)

    z = (resid / cond_vol).dropna()
    losses_z = -z
    eff_window = _safe_window(len(losses_z), window)

    qz_values = []
    idx = []
    last_fit = None
    failures = 0

    for i in range(eff_window, len(losses_z)):
        sample_losses_z = losses_z.iloc[i - eff_window:i]
        try:
            fit_info = _evt_pot_fit_on_losses(
                sample_losses_z,
                threshold_quantile=threshold_quantile,
                min_excess=min_excess,
            )
            q_loss_z = _evt_pot_quantile_from_fit(confidence, fit_info)
            qz_values.append(q_loss_z)
            idx.append(losses_z.index[i])
            last_fit = fit_info
        except Exception:
            failures += 1
            continue

    if not qz_values:
        raise ValueError(
            "Impossible d'estimer la VaR EVT-GARCH sur la fenêtre choisie. "
            "Le seuil EVT est probablement trop élevé ou les résidus standardisés ne présentent pas assez d'excès."
        )

    qz_series = pd.Series(qz_values, index=idx, name="qz_loss")
    sigma1_sq = (cond_vol.reindex(qz_series.index).shift(1).dropna() ** 2)
    qz_series = qz_series.reindex(sigma1_sq.index)

    sigma_h_sq = _aggregate_garch_variance(sigma1_sq, omega, alpha_g, beta_g, horizon)
    sigma_h = np.sqrt(sigma_h_sq)
    mu_h = horizon * mu_const

    # qz_series = quantile de perte positive des résidus standardisés
    var_series = sigma_h * qz_series - mu_h
    var_series = var_series.dropna()

    if var_series.empty:
        raise ValueError("Aucune prévision VaR EVT-GARCH disponible après reconstruction.")

    var_return = float(var_series.iloc[-1])

    diag = _portfolio_diagnostics(returns)
    diag.update(
        {
            "mean_model": mean_model,
            "lags": lags,
            "nu": nu,
            "omega": omega,
            "alpha_garch": alpha_g,
            "beta_garch": beta_g,
            "aic": float(fit.aic),
            "bic": float(fit.bic),
            "window": eff_window,
            "n_failed_windows": failures,
        }
    )
    if last_fit is not None:
        diag.update(last_fit)

    status = "warning" if failures > 0 else "success"
    message = (
        "VaR EVT-GARCH estimée avec succès."
        if failures == 0
        else "VaR EVT-GARCH estimée avec succès, avec quelques fenêtres non exploitables ignorées."
    )

    return VaRMethodResult(
        method="EVT-GARCH",
        category="Paramétrique",
        conditional=True,
        horizon=horizon,
        status=status,
        message=message,
        var_return=var_return,
        var_money=_return_var_to_money(var_return, st.session_state["capital_invested"]),
        var_series=var_series,
        diagnostics=diag,
    )


# ============================================================
# PIPELINE GLOBAL ROBUSTE
# ============================================================

def _run_method_safe(func, *args, **kwargs) -> VaRMethodResult:
    method_meta = kwargs.pop("_meta")
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return _make_error_result(
            method=method_meta["method"],
            category=method_meta["category"],
            conditional=method_meta["conditional"],
            horizon=method_meta["horizon"],
            message=str(e),
        )


def compute_all_var_methods(
    portfolio_prices_df: pd.DataFrame,
    weights: dict[str, float],
    confidence: float = 0.99,
    horizon: int = 1,
    capital_invested: float = 10000.0,
    rolling_window: int = 250,
) -> dict[str, Any]:
    """
    Pipeline principal robuste :
    - toutes les méthodes sont tentées
    - une méthode qui échoue n'interrompt pas l'ensemble
    - les diagnostics et statuts sont stockés proprement
    """
    series = build_portfolio_series(
        asset_price_df=portfolio_prices_df,
        weights=weights,
        capital_invested=capital_invested,
    )

    portfolio_value = _last_portfolio_value(series["portfolio_value"])
    returns = series["portfolio_returns"]

    methods = [
        (
            var_historical,
            {
                "method": "Historique",
                "category": "Non paramétrique",
                "conditional": False,
                "horizon": horizon,
            },
            {
                "returns": returns,
                "confidence": confidence,
                "horizon": horizon,
                "portfolio_value": portfolio_value,
                "window": rolling_window,
            },
        ),
        (
            var_variance_covariance,
            {
                "method": "Variance-Covariance",
                "category": "Semi-paramétrique",
                "conditional": True,
                "horizon": horizon,
            },
            {
                "returns": returns,
                "confidence": confidence,
                "horizon": horizon,
                "portfolio_value": portfolio_value,
                "window": rolling_window,
            },
        ),
        (
            var_cornish_fisher,
            {
                "method": "Cornish-Fisher",
                "category": "Semi-paramétrique",
                "conditional": True,
                "horizon": horizon,
            },
            {
                "returns": returns,
                "confidence": confidence,
                "horizon": horizon,
                "portfolio_value": portfolio_value,
                "window": rolling_window,
            },
        ),
        (
            var_riskmetrics,
            {
                "method": "RiskMetrics",
                "category": "Paramétrique",
                "conditional": True,
                "horizon": horizon,
            },
            {
                "returns": returns,
                "confidence": confidence,
                "horizon": horizon,
                "portfolio_value": portfolio_value,
            },
        ),
        (
            var_garch_student,
            {
                "method": "GARCH-Student",
                "category": "Paramétrique",
                "conditional": True,
                "horizon": horizon,
            },
            {
                "returns": returns,
                "confidence": confidence,
                "horizon": horizon,
                "portfolio_value": portfolio_value,
            },
        ),
        (
            var_evt,
            {
                "method": "EVT-POT",
                "category": "Semi-paramétrique",
                "conditional": True,
                "horizon": horizon,
            },
            {
                "returns": returns,
                "confidence": confidence,
                "horizon": horizon,
                "portfolio_value": portfolio_value,
                "window": rolling_window,
                "threshold_quantile": 0.95,
                "min_excess": None,
            },
        ),
        (
            var_evt_garch,
            {
                "method": "EVT-GARCH",
                "category": "Paramétrique",
                "conditional": True,
                "horizon": horizon,
            },
            {
                "returns": returns,
                "confidence": confidence,
                "horizon": horizon,
                "portfolio_value": portfolio_value,
                "window": rolling_window,
                "threshold_quantile": 0.95,
                "min_excess": None,
            },
        ),
    ]

    results: list[VaRMethodResult] = []
    for func, meta, kwargs in methods:
        result = _run_method_safe(func, _meta=meta, **kwargs)
        results.append(result)

    var_summary = pd.DataFrame(
        [
            {
                "Method": r.method,
                "Category": r.category,
                "Conditional": r.conditional,
                "Horizon": r.horizon,
                "Status": r.status,
                "Message": r.message,
                "VaR_Return": r.var_return,
                "VaR_Money": r.var_money,
            }
            for r in results
        ]
    )

    diagnostics = {r.method: r.diagnostics for r in results}
    var_series_dict = {
        r.method: r.var_series
        for r in results
        if r.status in {"success", "warning"} and r.var_series is not None
    }

    return {
        "portfolio_prices": series["portfolio_value"],
        "portfolio_returns": series["portfolio_returns"],
        "portfolio_pnl": series["portfolio_pnl"],
        "asset_prices": series["asset_prices"],
        "asset_log_returns": series["asset_log_returns"],
        "var_summary": var_summary,
        "diagnostics": diagnostics,
        "var_series_dict": var_series_dict,
        "confidence": confidence,
        "horizon": horizon,
        "capital_invested": capital_invested,
    }