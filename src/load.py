import os
import pandas as pd
import kagglehub

from config import (
    KAGGLE_DATASET_FRAMINGHAM, KAGGLE_DATASET_HEART,
    FRAMINGHAM_CACHE_PATH, HEART_CACHE_PATH,
    CARDIO_RENAME, FRAMINGHAM_RENAME, HEART_RENAME,
    HEART_SEX_MAP, COMMON_COLS, SOURCE_COL,
    SOURCE_CARDIO, SOURCE_FRAMINGHAM, SOURCE_HEART,
)

def load_cardio(csv_path: str) -> pd.DataFrame:
    """Load the Cardiovascular Disease Dataset from a local CSV file."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns=CARDIO_RENAME)
    df = df[COMMON_COLS].copy()
    df[SOURCE_COL] = SOURCE_CARDIO
    return df


def load_framingham() -> pd.DataFrame:
    """ Download the Framingham Heart Study dataset from kagglehub."""
    kagglehub.dataset_download(KAGGLE_DATASET_FRAMINGHAM)
    csv_path = os.path.expanduser(FRAMINGHAM_CACHE_PATH)
    df = pd.read_csv(csv_path)
    df = df.rename(columns=FRAMINGHAM_RENAME)
    df = df[COMMON_COLS].copy()
    df[SOURCE_COL] = SOURCE_FRAMINGHAM
    return df


def load_heart_failure() -> pd.DataFrame:
    """Download Heart Failure Prediction dataset from kagglehub."""
    kagglehub.dataset_download(KAGGLE_DATASET_HEART)
    csv_path = os.path.expanduser(HEART_CACHE_PATH)
    df = pd.read_csv(csv_path)
    df = df.rename(columns=HEART_RENAME)
    df["sex"] = df["sex"].map(HEART_SEX_MAP)
    df = df[COMMON_COLS].copy()
    df[SOURCE_COL] = SOURCE_HEART
    return df


def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame) -> pd.DataFrame:
    """drop NaNs and remove zero-cholesterol rows."""
    combined = pd.concat([df1, df2, df3], ignore_index=True)
    combined = combined.dropna()
    combined = combined[combined["cholesterol"] > 0]
    return combined


