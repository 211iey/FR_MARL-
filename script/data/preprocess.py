"""
Load excess_returns CSV, drop NaN warm-up rows, and split train/valid/test.

Returns a DataFrame trimmed to usable rows plus a column-group dict so that
the environment can pick per-asset feature channels without hard-coding names.
"""
import pandas as pd


ASSETS = [
    "semicon", "bank", "car", "finance", "treasury3", "ener&chem",
    "iron", "wti", "gold", "copper", "necessities", "healthcare",
    "silver", "treasury10",
]


def feature_columns(assets=ASSETS):
    """Per-asset feature column groups in (return, vol20, vol60) order."""
    return {
        "returns": list(assets),
        "vol20":   [f"vol20_{a}" for a in assets],
        "vol60":   [f"vol60_{a}" for a in assets],
        "market":  ["vkospi", "vol20_kospi"],
    }


def load_dataset(csv_path, assets=ASSETS):
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date").sort_index()
    cols = feature_columns(assets)
    needed = cols["returns"] + cols["vol20"] + cols["vol60"] + cols["market"]
    df = df[needed].dropna()  # trims vol60 warm-up + pre-2014 vkospi NaNs
    return df, cols


def split(df, train_end="2021-12-31", valid_end="2022-12-31"):
    train = df.loc[:train_end]
    valid = df.loc[train_end:valid_end].iloc[1:]
    test  = df.loc[valid_end:].iloc[1:]
    return {"train": train, "valid": valid, "test": test}


def normalizer(train_df, cols):
    """Compute per-column mean/std on the train split for z-score normalization."""
    feat_cols = cols["returns"] + cols["vol20"] + cols["vol60"] + cols["market"]
    mean = train_df[feat_cols].mean()
    std = train_df[feat_cols].std().replace(0, 1.0)
    return mean, std
