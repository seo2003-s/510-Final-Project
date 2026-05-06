
import argparse
import sys
import os

# Make sure src/ is on the path when running from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from load import load_cardio, load_framingham, load_heart_failure, merge_datasets
from analyze import (
    print_summary, plot_histograms, plot_boxplots, plot_scatter_pairs,
    split_and_scale, train_model, evaluate_model, check_multicollinearity,
    check_independence, plot_linearity_check,
    statsmodels_summary, odds_ratios,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Identifying Key Predictors of Cardiovascular Diseases")
    parser.add_argument(
        "--cardio-csv",
        required=True,
        help="Path to Cardiovascular_Disease_Dataset.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load datasets and merge them
    print("Loading datasets")
    df1 = load_cardio(args.cardio_csv)
    df2 = load_framingham()
    df3 = load_heart_failure()
    df  = merge_datasets(df1, df2, df3)
    print(f"Combined dataset shape: {df.shape}\n")

    # EDA
    print("Descriptive Statistics")
    print_summary(df)
    plot_histograms(df)
    plot_boxplots(df)
    plot_scatter_pairs(df)

    # Split and scale first
    X_train_s, X_test_s, X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    # Assumption checks
    print("\nMulticollinearity Check")
    check_multicollinearity(X_train_s)

    # Model training
    print("\nTraining Logistic Regression")
    model = train_model(X_train_s, y_train)

    # Model Evaluation
    print("\nModel Evaluation")
    evaluate_model(model, X_test_s, y_test)

    # Assumption checks
    
    print("\nIndependence Check")
    check_independence(model, X_train_s, y_train)

    print("\nLinearity Check")
    plot_linearity_check(model, X_train_s, X_train)

    # Model summary
    print("\nStatsmodels Summary")
    sm_model = statsmodels_summary(X_train_s, y_train)

    print("\nOdds Ratios")
    odds_ratios(sm_model)



if __name__ == "__main__":
    main()