from itertools import combinations

import numpy as np
import pandas as pd


def compute_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()

    if len(aligned) < 20:
        return np.nan

    asset = aligned.iloc[:, 0]
    market = aligned.iloc[:, 1]

    var_market = np.var(market, ddof=1)
    if pd.isna(var_market) or var_market <= 0:
        return np.nan

    cov = np.cov(asset, market, ddof=1)[0, 1]
    return float(cov / var_market)


def compute_all_betas(asset_returns: pd.DataFrame, market_returns: pd.Series) -> pd.Series:
    betas = {asset: compute_beta(asset_returns[asset], market_returns) for asset in asset_returns.columns}
    return pd.Series(betas, name="Beta")


def expected_returns_capm(
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate_annual: float = 0.02,
    annualization_factor: int = 252,
) -> pd.Series:
    betas = compute_all_betas(asset_returns, market_returns)
    market_return_annual = market_returns.mean() * annualization_factor
    expected = risk_free_rate_annual + betas * (market_return_annual - risk_free_rate_annual)
    expected.name = "Expected_Return_CAPM"
    return expected


def compute_portfolio_weights(tickers: list[str]) -> pd.Series:
    n = len(tickers)
    return pd.Series(np.repeat(1 / n, n), index=tickers)


def compute_portfolio_metrics(
    tickers: list[str],
    asset_returns: pd.DataFrame,
    expected_returns: pd.Series,
    betas: pd.Series,
    annualization_factor: int = 252,
):
    sub_returns = asset_returns[tickers].copy()

    # on conserve les dates exploitables
    sub_returns = sub_returns.dropna(how="any")

    if len(sub_returns) < 20:
        raise ValueError("Historique insuffisant.")

    sub_expected = expected_returns[tickers]
    sub_betas = betas[tickers]

    if sub_expected.isna().any():
        raise ValueError("Rendements CAPM invalides.")

    if sub_betas.isna().any():
        raise ValueError("Bêtas invalides.")

    weights = compute_portfolio_weights(tickers)

    cov_matrix = sub_returns.cov() * annualization_factor
    if cov_matrix.isna().any().any():
        raise ValueError("Covariance invalide.")

    w = weights.values.reshape(-1, 1)
    port_return = float(np.dot(weights.values, sub_expected.values))
    port_vol = float(np.sqrt(np.dot(np.dot(w.T, cov_matrix.values), w)).item())
    port_beta = float(np.dot(weights.values, sub_betas.values))

    if not np.isfinite(port_vol) or port_vol <= 0:
        raise ValueError("Volatilité invalide.")

    score = float(port_return / port_vol)

    return {
        "assets": tickers,
        "weights": weights.round(4).to_dict(),
        "expected_return": port_return,
        "volatility": port_vol,
        "beta": port_beta,
        "score": score,
        "concentration": "Moyenne",
    }


def generate_candidate_portfolios(
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    n_assets: int = 5,
    annualization_factor: int = 252,
    risk_free_rate_annual: float = 0.02,
) -> pd.DataFrame:
    tickers = list(asset_returns.columns)

    if len(tickers) < n_assets:
        raise ValueError(f"Seulement {len(tickers)} actifs disponibles, minimum requis : {n_assets}")

    expected_returns = expected_returns_capm(
        asset_returns=asset_returns,
        market_returns=market_returns,
        risk_free_rate_annual=risk_free_rate_annual,
        annualization_factor=annualization_factor,
    )

    betas = compute_all_betas(asset_returns, market_returns)

    valid_assets = [
        t for t in tickers
        if pd.notna(expected_returns.get(t)) and pd.notna(betas.get(t)) and asset_returns[t].notna().sum() >= 20
    ]

    if len(valid_assets) < n_assets:
        raise ValueError(
            f"Actifs valides insuffisants après filtrage : {len(valid_assets)}"
        )

    candidates = []
    for combo in combinations(valid_assets, n_assets):
        try:
            metrics = compute_portfolio_metrics(
                tickers=list(combo),
                asset_returns=asset_returns,
                expected_returns=expected_returns,
                betas=betas,
                annualization_factor=annualization_factor,
            )
            candidates.append(metrics)
        except Exception:
            continue

    if not candidates:
        raise ValueError(
            "Aucun portefeuille candidat valide n'a été généré. "
            "Les rendements ou les bêtas restent probablement invalides."
        )

    df = pd.DataFrame(candidates)
    df = df.sort_values(["score", "expected_return"], ascending=[False, False]).reset_index(drop=True)
    return df


def get_top_4_portfolio_suggestions(
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    annualization_factor: int = 252,
    risk_free_rate_annual: float = 0.02,
) -> pd.DataFrame:
    all_candidates = generate_candidate_portfolios(
        asset_returns=asset_returns,
        market_returns=market_returns,
        n_assets=5,
        annualization_factor=annualization_factor,
        risk_free_rate_annual=risk_free_rate_annual,
    )

    top4 = all_candidates.head(4).copy()

    labels = [
        ("Portefeuille Défensif", "Allocation prudente"),
        ("Portefeuille Équilibré", "Compromis rendement / risque"),
        ("Portefeuille Dynamique", "Recherche de rendement"),
        ("Portefeuille Opportunité", "Sélection CAPM"),
    ]

    top4["portfolio_name"] = [labels[i][0] for i in range(len(top4))]
    top4["profile"] = [labels[i][1] for i in range(len(top4))]

    return top4

def compute_capm_inputs(
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate_annual: float = 0.02,
    annualization_factor: int = 252,
):
    """
    Retourne les éléments de base du modèle CAPM
    utiles pour l'optimisation.
    """
    betas = compute_all_betas(asset_returns, market_returns)
    expected_returns = expected_returns_capm(
        asset_returns=asset_returns,
        market_returns=market_returns,
        risk_free_rate_annual=risk_free_rate_annual,
        annualization_factor=annualization_factor,
    )
    cov_matrix = asset_returns.cov() * annualization_factor

    return expected_returns, cov_matrix, betas


def compute_equal_weights(selected_assets: list[str]) -> dict:
    """
    Stratégie 1 : portefeuille équilibré
    """
    if len(selected_assets) == 0:
        raise ValueError("Aucun actif sélectionné.")

    n = len(selected_assets)
    w = 1.0 / n
    return {asset: w for asset in selected_assets}


def compute_custom_weights_from_percent(
    selected_assets: list[str],
    entered_weights: dict
) -> dict:
    """
    Stratégie 2 : portefeuille personnalisé
    Les poids saisis par l'utilisateur sont en %.
    """
    if len(selected_assets) == 0:
        raise ValueError("Aucun actif sélectionné.")

    weights = {}
    total = 0.0

    for asset in selected_assets:
        value = float(entered_weights.get(asset, 0.0))
        weights[asset] = value / 100.0
        total += value

    if abs(total - 100.0) > 1e-6:
        raise ValueError("La somme des pondérations doit être égale à 100 %.")

    return weights


def compute_portfolio_performance_from_weights(
    selected_assets: list[str],
    weights: dict,
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate_annual: float = 0.02,
    annualization_factor: int = 252,
) -> dict:
    """
    Calcule le rendement attendu, la volatilité et le bêta
    d'un portefeuille défini par ses poids.
    """
    if len(selected_assets) == 0:
        raise ValueError("Aucun actif sélectionné.")

    sub_returns = asset_returns[selected_assets].dropna(how="any")
    if len(sub_returns) < 20:
        raise ValueError("Historique insuffisant pour les actifs sélectionnés.")

    aligned_market = market_returns.loc[sub_returns.index]

    expected_returns, cov_matrix, betas = compute_capm_inputs(
        asset_returns=sub_returns,
        market_returns=aligned_market,
        risk_free_rate_annual=risk_free_rate_annual,
        annualization_factor=annualization_factor,
    )

    weight_vector = np.array([weights[a] for a in selected_assets], dtype=float)
    mu = expected_returns[selected_assets].values
    beta_vec = betas[selected_assets].values
    cov = cov_matrix.loc[selected_assets, selected_assets].values

    port_return = float(np.dot(weight_vector, mu))
    port_vol = float(np.sqrt(np.dot(weight_vector.T, np.dot(cov, weight_vector))))
    port_beta = float(np.dot(weight_vector, beta_vec))
    score = float(port_return / port_vol) if port_vol > 0 else np.nan

    return {
        "expected_return": port_return,
        "volatility": port_vol,
        "beta": port_beta,
        "score": score,
    }


def compute_capm_optimized_weights(
    selected_assets: list[str],
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    target_return: float | None = None,
    target_volatility: float | None = None,
    risk_free_rate_annual: float = 0.02,
    annualization_factor: int = 252,
    n_random: int = 20000,
    tolerance: float = 0.01,
) -> dict:
    """
    Optimisation CAPM long-only par simulation.

    - Si target_return est renseigné :
      on cherche le portefeuille le plus proche du rendement cible.

    - Si target_volatility est renseigné :
      on cherche le portefeuille le plus proche de la volatilité cible.

    - Si aucune cible n'est renseignée :
      on maximise le ratio rendement / risque.

    tolerance = 0.01 signifie 1 point de pourcentage.
    """
    if len(selected_assets) == 0:
        raise ValueError("Aucun actif sélectionné.")

    sub_returns = asset_returns[selected_assets].dropna(how="any")
    if len(sub_returns) < 20:
        raise ValueError("Historique insuffisant pour l’optimisation.")

    aligned_market = market_returns.loc[sub_returns.index]

    expected_returns, cov_matrix, betas = compute_capm_inputs(
        asset_returns=sub_returns,
        market_returns=aligned_market,
        risk_free_rate_annual=risk_free_rate_annual,
        annualization_factor=annualization_factor,
    )

    mu = expected_returns[selected_assets].values
    cov = cov_matrix.loc[selected_assets, selected_assets].values
    beta_vec = betas[selected_assets].values

    if np.isnan(mu).any():
        raise ValueError("Les rendements attendus CAPM contiennent des NaN.")
    if np.isnan(cov).any():
        raise ValueError("La matrice de covariance contient des NaN.")
    if np.isnan(beta_vec).any():
        raise ValueError("Les bêtas contiennent des NaN.")

    # Bornes réalistes long-only
    min_return = float(np.min(mu))
    max_return = float(np.max(mu))

    asset_vols = np.sqrt(np.diag(cov))
    min_vol = float(np.min(asset_vols))
    max_vol = float(np.max(asset_vols))

    if target_return is not None:
        if target_return < min_return or target_return > max_return:
            raise ValueError(
                f"Rendement cible irréaliste. Avec les actifs sélectionnés, "
                f"le rendement attendu faisable est compris entre {min_return:.2%} et {max_return:.2%}."
            )

    if target_volatility is not None:
        if target_volatility < min_vol or target_volatility > max_vol:
            raise ValueError(
                f"Volatilité cible irréaliste. Avec les actifs sélectionnés, "
                f"la volatilité faisable est comprise entre {min_vol:.2%} et {max_vol:.2%}."
            )

    best_result = None
    best_loss = np.inf

    for _ in range(n_random):
        w = np.random.random(len(selected_assets))
        w = w / w.sum()

        port_return = float(np.dot(w, mu))
        port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
        port_beta = float(np.dot(w, beta_vec))

        if not np.isfinite(port_vol) or port_vol <= 0:
            continue

        if target_return is not None:
            loss = abs(port_return - target_return)
        elif target_volatility is not None:
            loss = abs(port_vol - target_volatility)
        else:
            loss = -(port_return / port_vol)

        if loss < best_loss:
            best_loss = loss
            best_result = {
                "weights": {
                    asset: float(weight) for asset, weight in zip(selected_assets, w)
                },
                "expected_return": port_return,
                "volatility": port_vol,
                "beta": port_beta,
                "score": float(port_return / port_vol),
            }

    if best_result is None:
        raise ValueError("Aucune solution d’optimisation valide n’a été trouvée.")

    best_result["target_met"] = True
    best_result["message"] = "Optimisation CAPM réalisée avec succès."
    best_result["target_gap"] = 0.0

    if target_return is not None:
        gap = abs(best_result["expected_return"] - target_return)
        best_result["target_gap"] = gap
        if gap > tolerance:
            best_result["target_met"] = False
            best_result["message"] = (
                f"La cible de rendement n’a pas pu être atteinte exactement. "
                f"Le portefeuille affiché est le plus proche trouvé, avec un écart de {gap:.2%}."
            )

    if target_volatility is not None:
        gap = abs(best_result["volatility"] - target_volatility)
        best_result["target_gap"] = gap
        if gap > tolerance:
            best_result["target_met"] = False
            best_result["message"] = (
                f"La cible de volatilité n’a pas pu être atteinte exactement. "
                f"Le portefeuille affiché est le plus proche trouvé, avec un écart de {gap:.2%}."
            )

    return best_result