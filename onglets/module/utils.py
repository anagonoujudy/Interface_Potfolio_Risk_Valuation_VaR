import pandas as pd
import numpy as np

INDEX_COLUMN = "Cac40"


def _clean_column_name(col: str) -> str:
    return str(col).strip()


def _parse_french_number(x):
    """
    Convertit proprement les nombres au format français :
    - 554,7
    - 8,515.49
    - 8 515,49
    """
    if pd.isna(x):
        return pd.NA

    if isinstance(x, (int, float)):
        return x

    s = str(x).strip().replace("\u202f", "").replace(" ", "")

    if s == "":
        return pd.NA

    # Cas 1 : contient virgule et point
    # on suppose que le dernier séparateur rencontré est le décimal
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            # format type 8.515,49
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            # format type 8,515.49
            s = s.replace(",", "")
    elif "," in s:
        # format type 554,7
        s = s.replace(",", ".")
    else:
        # format type 554.7 ou 8515
        pass

    try:
        return float(s)
    except ValueError:
        return pd.NA


def load_price_data(path: str = "Data_set.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path, dtype=object)

    df.columns = [_clean_column_name(c) for c in df.columns]

    if "Date" not in df.columns:
        raise ValueError(f"Colonne 'Date' absente. Colonnes détectées : {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    numeric_cols = [col for col in df.columns if col != "Date"]

    for col in numeric_cols:
        df[col] = df[col].apply(_parse_french_number)

    # On supprime les colonnes complètement vides
    df = df.dropna(axis=1, how="all")

    # Vérifie qu'il reste bien l'indice
    if INDEX_COLUMN not in df.columns:
        raise ValueError(
            f"La colonne d'indice '{INDEX_COLUMN}' est absente après nettoyage. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    # On supprime les lignes où tout est vide côté numérique
    existing_numeric_cols = [c for c in df.columns if c != "Date"]
    df = df.dropna(how="all", subset=existing_numeric_cols)

    df = df.set_index("Date")

    return df


def get_index_series(price_df: pd.DataFrame, index_col: str = INDEX_COLUMN) -> pd.Series:
    if index_col not in price_df.columns:
        raise ValueError(
            f"La colonne d'indice '{index_col}' est absente. "
            f"Colonnes disponibles : {list(price_df.columns)}"
        )

    series = price_df[index_col].dropna()
    if series.empty:
        raise ValueError(f"La série de l'indice '{index_col}' est vide.")
    return series


def get_asset_prices(price_df: pd.DataFrame, index_col: str = INDEX_COLUMN) -> pd.DataFrame:
    asset_cols = [col for col in price_df.columns if col != index_col]
    if not asset_cols:
        raise ValueError("Aucune action détectée dans la base.")
    asset_df = price_df[asset_cols].copy()

    # on enlève les colonnes d'actifs totalement vides
    asset_df = asset_df.dropna(axis=1, how="all")

    if asset_df.empty:
        raise ValueError("Toutes les colonnes d'actions sont vides après nettoyage.")

    return asset_df


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(price_df / price_df.shift(1))
    returns = returns.dropna()
    return returns


def split_asset_and_market_returns(price_df: pd.DataFrame, index_col: str = INDEX_COLUMN):
    asset_prices = get_asset_prices(price_df, index_col=index_col)
    market_prices = get_index_series(price_df, index_col=index_col)

    asset_returns = compute_returns(asset_prices)
    market_returns = compute_returns(market_prices.to_frame()).iloc[:, 0]

    aligned = asset_returns.join(market_returns.rename(index_col), how="inner")
    aligned = aligned.dropna(how="all")

    if aligned.empty:
        raise ValueError("Aucune donnée alignée disponible après calcul des rendements.")

    asset_returns = aligned.drop(columns=[index_col])
    market_returns = aligned[index_col]

    # on supprime les actifs trop incomplets
    valid_cols = asset_returns.columns[asset_returns.notna().sum() >= 20]
    asset_returns = asset_returns[valid_cols]

    if asset_returns.shape[1] < 5:
        raise ValueError(
            f"Nombre d'actions exploitables insuffisant après nettoyage : {asset_returns.shape[1]}"
        )

    if market_returns.dropna().shape[0] < 20:
        raise ValueError("Historique de marché insuffisant après nettoyage.")

    return asset_returns, market_returns