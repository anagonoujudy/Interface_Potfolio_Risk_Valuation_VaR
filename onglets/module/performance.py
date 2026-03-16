from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def compute_portfolio_market_series(
    asset_price_df: pd.DataFrame,
    weights: dict[str, float],
    market_series: pd.Series | None = None,
    capital_invested: float = 10000.0,
):
    """
    Construit :
    - série de valeur du portefeuille
    - rendements log du portefeuille
    - PnL historique
    - série de marché normalisée sur le même capital
    """
    weights_s = pd.Series(weights, dtype=float)
    selected_assets = list(weights_s.index)

    prices = asset_price_df[selected_assets].dropna(how="any").copy()
    if prices.empty:
        raise ValueError("Aucune donnée de prix disponible pour le portefeuille.")

    norm_prices = prices / prices.iloc[0]
    portfolio_base1 = norm_prices.mul(weights_s, axis=1).sum(axis=1)
    portfolio_value = portfolio_base1 * capital_invested

    portfolio_returns = np.log(portfolio_value / portfolio_value.shift(1)).dropna()
    portfolio_pnl = portfolio_value - capital_invested

    market_value = None
    market_returns = None
    if market_series is not None:
        aligned_market = market_series.loc[portfolio_value.index].dropna()
        if not aligned_market.empty:
            aligned_market = aligned_market / aligned_market.iloc[0] * capital_invested
            market_value = aligned_market
            market_returns = np.log(aligned_market / aligned_market.shift(1)).dropna()

    return {
        "portfolio_value": portfolio_value,
        "portfolio_returns": portfolio_returns,
        "portfolio_pnl": portfolio_pnl,
        "market_value": market_value,
        "market_returns": market_returns,
    }


def compute_drawdown(portfolio_value: pd.Series) -> pd.Series:
    running_max = portfolio_value.cummax()
    drawdown = portfolio_value / running_max - 1.0
    return drawdown


def compute_performance_metrics(
    portfolio_value: pd.Series,
    portfolio_returns: pd.Series,
    market_returns: pd.Series | None = None,
    risk_free_rate_annual: float = 0.02,
):
    if portfolio_returns.empty:
        raise ValueError("Série de rendements portefeuille vide.")

    total_return = float(portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1.0)

    n_days = len(portfolio_returns)
    annualized_return = float(np.exp(portfolio_returns.mean() * TRADING_DAYS) - 1.0)
    annualized_vol = float(portfolio_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))

    rf_daily_log = np.log(1 + risk_free_rate_annual) / TRADING_DAYS
    excess_returns = portfolio_returns - rf_daily_log
    sharpe = float((excess_returns.mean() / portfolio_returns.std(ddof=1)) * np.sqrt(TRADING_DAYS)) if portfolio_returns.std(ddof=1) > 0 else np.nan

    downside = portfolio_returns[portfolio_returns < rf_daily_log]
    downside_std = downside.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(downside) > 1 else np.nan
    sortino = float((annualized_return - risk_free_rate_annual) / downside_std) if downside_std and downside_std > 0 else np.nan

    drawdown = compute_drawdown(portfolio_value)
    max_drawdown = float(drawdown.min())

    beta = np.nan
    corr = np.nan
    if market_returns is not None and not market_returns.empty:
        aligned = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
        if len(aligned) > 10:
            rp = aligned.iloc[:, 0]
            rm = aligned.iloc[:, 1]
            var_m = np.var(rm, ddof=1)
            if var_m > 0:
                beta = float(np.cov(rp, rm, ddof=1)[0, 1] / var_m)
            corr = float(rp.corr(rm))

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "Beta vs Market": beta,
        "Correlation vs Market": corr,
        "Observations": n_days,
    }


def compute_asset_contributions(
    asset_price_df: pd.DataFrame,
    weights: dict[str, float],
):
    weights_s = pd.Series(weights, dtype=float)
    prices = asset_price_df[weights_s.index].dropna(how="any").copy()
    returns = np.log(prices / prices.shift(1)).dropna()

    annual_ret = np.exp(returns.mean() * TRADING_DAYS) - 1.0
    annual_vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

    df = pd.DataFrame({
        "Weight": weights_s,
        "Annualized Return": annual_ret,
        "Annualized Volatility": annual_vol,
    })
    df["Weighted Return Contribution"] = df["Weight"] * df["Annualized Return"]
    return df.reset_index().rename(columns={"index": "Asset"})


def monte_carlo_pnl_forecast(
    portfolio_returns: pd.Series,
    current_portfolio_value: float,
    horizon_days: int = 16,
    n_sims: int = 5000,
    seed: int = 42,
):
    """
    Prévision simple de PnL sous hypothèse log-normale :
    r_t ~ N(mu, sigma²)
    """
    if portfolio_returns.empty:
        raise ValueError("Série de rendements vide pour Monte Carlo.")

    mu = float(portfolio_returns.mean())
    sigma = float(portfolio_returns.std(ddof=1))

    rng = np.random.default_rng(seed)
    sims = rng.normal(loc=mu, scale=sigma, size=(horizon_days, n_sims))
    cum_log_returns = np.cumsum(sims, axis=0)

    forecast_values = current_portfolio_value * np.exp(cum_log_returns)
    forecast_pnl = forecast_values - current_portfolio_value

    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    forecast_quantiles = pd.DataFrame(
        {f"q{int(q*100):02d}": np.quantile(forecast_pnl, q, axis=1) for q in quantiles},
        index=pd.RangeIndex(1, horizon_days + 1, name="Day"),
    )

    return {
        "mu_daily": mu,
        "sigma_daily": sigma,
        "forecast_values": forecast_values,
        "forecast_pnl": forecast_pnl,
        "forecast_quantiles": forecast_quantiles,
    }