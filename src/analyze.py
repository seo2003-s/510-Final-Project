import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from config import (
    FEATURE_COLS, TARGET_COL, NUMERIC_COLS,
    TEST_SIZE, RANDOM_STATE, DECISION_THRESHOLD, MAX_ITER,
    PLOT_LINEARITY,
)


# ── EDA ──
def print_summary(df: pd.DataFrame) -> None:
    """Prints descriptive statistics for key predictors."""
    print("Means\n", df[NUMERIC_COLS].mean())
    print("\nMin\n", df[NUMERIC_COLS].min())
    print("\nMax\n", df[NUMERIC_COLS].max())
    print("\nFull Summary\n", df[NUMERIC_COLS].describe())
    iqr = df[NUMERIC_COLS].quantile(0.75) - df[NUMERIC_COLS].quantile(0.25)
    print("\nIQR\n", iqr)
    print("\nSex value counts\n", df["sex"].value_counts())
    print(f"\nDataset shape: {df.shape}")


def plot_histograms(df: pd.DataFrame) -> None:
    """Plots histograms for key predictors."""
    df[NUMERIC_COLS].hist(figsize=(10, 8))
    plt.tight_layout()
    plt.show()


def plot_boxplots(df: pd.DataFrame) -> None:
    """Plot boxplots for  key predictors."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[NUMERIC_COLS])
    plt.xticks(rotation=45)
    plt.title("Boxplot of Key Variables")
    plt.tight_layout()
    plt.show()


def plot_scatter_pairs(df: pd.DataFrame) -> None:
    """Plot scatter plots for key variable pairs."""
    pairs = [
        ("age",        "cholesterol", "Age vs Cholesterol"),
        ("age",        "resting_bp",  "Age vs Resting BP"),
        ("resting_bp", "cholesterol", "Resting BP vs Cholesterol"),
    ]
    for x_col, y_col, title in pairs:
        plt.figure()
        sns.scatterplot(x=x_col, y=y_col, data=df)
        plt.title(title)
        plt.tight_layout()
        plt.show()


# ── Model training ──
def split_and_scale(df: pd.DataFrame):
    """Split into test sets and apply StandardScaler."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler


def train_model(X_train_scaled: np.ndarray, y_train: pd.Series) -> LogisticRegression:
    """logistic regression model with class weights."""
    model = LogisticRegression(class_weight="balanced", max_iter=MAX_ITER)
    model.fit(X_train_scaled, y_train)
    return model


def evaluate_model(
    model: LogisticRegression,
    X_test_scaled: np.ndarray,
    y_test: pd.Series,
    threshold: float = DECISION_THRESHOLD,
) -> dict:
    """Evaluate model and print classification report and key metrics."""
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    y_pred  = (y_probs > threshold).astype(int)

    print(classification_report(y_test, y_pred))
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"AUC:       {roc_auc_score(y_test, y_probs):.4f}")

    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "auc":       roc_auc_score(y_test, y_probs),
    }

# ── Assumption checks ──
def check_multicollinearity(X_train_scaled: np.ndarray) -> pd.DataFrame:
    """Finds VIF values for multicollinearity check"""
    vif_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "VIF": [
            variance_inflation_factor(X_train_scaled, i)
            for i in range(X_train_scaled.shape[1])
        ],
    })
    print(vif_df)
    return vif_df


def check_independence(
    model: LogisticRegression,
    X_train_scaled: np.ndarray,
    y_train: pd.Series,) -> float:
    """Additional Durbin Watson test for independence"""
    y_pred_prob = model.predict_proba(X_train_scaled)[:, 1]
    residuals   = y_train - y_pred_prob
    dw = durbin_watson(residuals)
    print(f"Durbin-Watson: {dw:.4f}")
    return dw


def plot_linearity_check(
    model: LogisticRegression,
    X_train_scaled: np.ndarray,
    X_train: pd.DataFrame,
    save_path: str = PLOT_LINEARITY,) -> None:
    """Plots each predictor against the log-odds to check linearity."""
    y_pred_prob = model.predict_proba(X_train_scaled)[:, 1]
    logit = np.log(y_pred_prob / (1 - y_pred_prob))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ["age", "cholesterol", "resting_bp"]):
        ax.scatter(X_train[col], logit, alpha=0.3, s=10)
        z      = np.polyfit(X_train[col], logit, 1)
        p      = np.poly1d(z)
        x_line = np.linspace(X_train[col].min(), X_train[col].max(), 100)
        ax.plot(x_line, p(x_line), color="red", linewidth=2)
        ax.set_xlabel(col)
        ax.set_ylabel("Log-odds (logit)")
        ax.set_title(f"{col} vs Logit")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def statsmodels_summary(X_train_scaled: np.ndarray, y_train: pd.Series):
    """Fit Logit and print full summary."""
    X_train_const = sm.add_constant(X_train_scaled)
    sm_model = sm.Logit(y_train, X_train_const).fit()
    print(sm_model.summary())
    return sm_model


def odds_ratios(sm_model) -> pd.DataFrame:
    """Compute odds ratios and conf intervals from the logit model."""
    or_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "odds_ratio": np.exp(sm_model.params[1:]),
        "lower_ci": np.exp(sm_model.conf_int()[0][1:]),
        "upper_ci": np.exp(sm_model.conf_int()[1][1:]),
    }).sort_values("odds_ratio", ascending=False)
    print(or_df)
    return or_df