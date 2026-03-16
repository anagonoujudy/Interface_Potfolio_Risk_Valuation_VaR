from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def _to_series(var_obj, index: pd.Index) -> pd.Series:
    if np.isscalar(var_obj):
        return pd.Series(float(var_obj), index=index)

    if isinstance(var_obj, pd.DataFrame):
        if var_obj.shape[1] != 1:
            raise ValueError("La VaR DataFrame doit contenir une seule colonne.")
        var_obj = var_obj.iloc[:, 0]

    if isinstance(var_obj, pd.Series):
        return var_obj.reindex(index).ffill().bfill()

    raise TypeError("Type de VaR non supporté.")


def compute_forward_horizon_returns(returns: pd.Series, horizon: int) -> pd.Series:
    """
    Rendement futur sur h jours aligné à t :
    r_{t,t+h} = r_t + ... + r_{t+h-1}
    """
    if horizon < 1:
        raise ValueError("L'horizon doit être >= 1.")

    if horizon == 1:
        return returns.dropna()

    out = pd.Series(index=returns.index, dtype=float)
    values = returns.values
    for i in range(len(values) - horizon + 1):
        out.iloc[i] = float(np.sum(values[i:i + horizon]))
    return out.dropna()


def compute_violations(
    returns: pd.Series,
    var_obj,
    horizon: int = 1,
) -> pd.Series:
    """
    Violation si le rendement futur sur h jours est inférieur à -VaR_t.
    """
    realized = compute_forward_horizon_returns(returns.dropna(), horizon)
    var_series = _to_series(var_obj, realized.index)

    aligned = pd.concat(
        [realized.rename("realized"), var_series.rename("var")],
        axis=1
    ).dropna()

    violations = aligned["realized"] < -aligned["var"]
    return violations.astype(int)


def kupiec_test(violations: pd.Series, confidence: float = 0.99) -> dict:
    v = violations.astype(int).values
    n = len(v)
    x = int(v.sum())

    if n == 0:
        raise ValueError("Série de violations vide.")

    p = 1.0 - confidence
    phat = x / n

    eps = 1e-12
    phat = np.clip(phat, eps, 1 - eps)

    lr_uc = -2.0 * np.log(
        (((1 - p) ** (n - x)) * (p ** x)) /
        (((1 - phat) ** (n - x)) * (phat ** x))
    )

    p_value = 1.0 - chi2.cdf(lr_uc, df=1)

    return {
        "LR_uc": float(lr_uc),
        "p_value": float(p_value),
        "n_obs": int(n),
        "n_violations": int(x),
        "expected_violations": float(n * p),
        "violation_ratio": float(x / n),
    }


def christoffersen_independence_test(violations: pd.Series) -> dict:
    v = violations.astype(int).values
    if len(v) < 2:
        raise ValueError("Série trop courte pour le test de Christoffersen.")

    n00 = n01 = n10 = n11 = 0

    for i in range(1, len(v)):
        if v[i - 1] == 0 and v[i] == 0:
            n00 += 1
        elif v[i - 1] == 0 and v[i] == 1:
            n01 += 1
        elif v[i - 1] == 1 and v[i] == 0:
            n10 += 1
        elif v[i - 1] == 1 and v[i] == 1:
            n11 += 1

    eps = 1e-12

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else eps
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else eps
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else eps

    pi0 = np.clip(pi0, eps, 1 - eps)
    pi1 = np.clip(pi1, eps, 1 - eps)
    pi = np.clip(pi, eps, 1 - eps)

    num = ((1 - pi) ** (n00 + n10)) * (pi ** (n01 + n11))
    den = ((1 - pi0) ** n00) * (pi0 ** n01) * ((1 - pi1) ** n10) * (pi1 ** n11)

    lr_ind = -2.0 * np.log(num / den)
    p_value = 1.0 - chi2.cdf(lr_ind, df=1)

    return {
        "LR_ind": float(lr_ind),
        "p_value": float(p_value),
        "n00": int(n00),
        "n01": int(n01),
        "n10": int(n10),
        "n11": int(n11),
    }


def christoffersen_conditional_coverage_test(
    violations: pd.Series,
    confidence: float = 0.99,
) -> dict:
    kupiec = kupiec_test(violations, confidence=confidence)
    indep = christoffersen_independence_test(violations)

    lr_cc = kupiec["LR_uc"] + indep["LR_ind"]
    p_value = 1.0 - chi2.cdf(lr_cc, df=2)

    return {
        "LR_cc": float(lr_cc),
        "p_value": float(p_value),
    }


def backtest_single_var(
    returns: pd.Series,
    var_obj,
    confidence: float = 0.99,
    horizon: int = 1,
    method_name: str = "VaR",
) -> dict:
    violations = compute_violations(
        returns=returns,
        var_obj=var_obj,
        horizon=horizon,
    )

    kupiec = kupiec_test(violations, confidence=confidence)
    indep = christoffersen_independence_test(violations)
    cc = christoffersen_conditional_coverage_test(violations, confidence=confidence)

    return {
        "Method": method_name,
        "Observations": kupiec["n_obs"],
        "Violations": kupiec["n_violations"],
        "Expected Violations": kupiec["expected_violations"],
        "Violation Ratio": kupiec["violation_ratio"],
        "Kupiec LR": kupiec["LR_uc"],
        "Kupiec p-value": kupiec["p_value"],
        "Christoffersen LR": indep["LR_ind"],
        "Christoffersen p-value": indep["p_value"],
        "Conditional Coverage LR": cc["LR_cc"],
        "Conditional Coverage p-value": cc["p_value"],
        "Violations Series": violations,
    }


def backtest_var_dict(
    returns: pd.Series,
    var_dict: dict,
    confidence: float = 0.99,
    horizon: int = 1,
) -> tuple[dict, pd.DataFrame]:
    detailed = {}

    for method_name, var_obj in var_dict.items():
        try:
            detailed[method_name] = backtest_single_var(
                returns=returns,
                var_obj=var_obj,
                confidence=confidence,
                horizon=horizon,
                method_name=method_name,
            )
        except Exception as e:
            detailed[method_name] = {"Method": method_name, "Error": str(e)}

    rows = []
    for method_name, result in detailed.items():
        if "Error" in result:
            rows.append({"Method": method_name, "Error": result["Error"]})
        else:
            rows.append({
                "Method": result["Method"],
                "Observations": result["Observations"],
                "Violations": result["Violations"],
                "Expected Violations": result["Expected Violations"],
                "Violation Ratio": result["Violation Ratio"],
                "Kupiec p-value": result["Kupiec p-value"],
                "Christoffersen p-value": result["Christoffersen p-value"],
                "Conditional Coverage p-value": result["Conditional Coverage p-value"],
            })

    summary_df = pd.DataFrame(rows)
    return detailed, summary_df


def select_best_var_model(backtest_summary: pd.DataFrame) -> pd.Series:
    df = backtest_summary.copy()

    for col in ["Method", "Kupiec p-value", "Conditional Coverage p-value"]:
        if col not in df.columns:
            raise ValueError(f"Colonne absente du tableau de backtesting : {col}")

    df = df.dropna(subset=["Kupiec p-value", "Conditional Coverage p-value"])

    if df.empty:
        raise ValueError("Aucun modèle de VaR valide à sélectionner.")

    df = df.sort_values(
        by=["Kupiec p-value", "Conditional Coverage p-value"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return df.iloc[0]