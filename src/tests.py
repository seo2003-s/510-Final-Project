import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from load import merge_datasets
from analyze import split_and_scale, train_model, evaluate_model, check_multicollinearity
from config import FEATURE_COLS

def _make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age":         rng.integers(30, 80, n).astype(float),
        "sex":         rng.integers(0, 2, n).astype(float),
        "cholesterol": rng.integers(150, 350, n).astype(float),
        "resting_bp":  rng.integers(80, 180, n).astype(float),
        "target":      rng.integers(0, 2, n),
        "source":      rng.choice(["cardio", "framingham", "heart"], n),
    })

class TestMergeDatasets(unittest.TestCase):

    def test_merge_drops_nan(self):
        df1 = _make_df(100, seed=1)
        df2 = _make_df(100, seed=2)
        df2.loc[0, "cholesterol"] = np.nan
        merged = merge_datasets(df1, df2, _make_df(100, seed=3))
        self.assertFalse(merged.isnull().any().any())

    def test_merge_drops_zero_cholesterol(self):
        df1 = _make_df(100, seed=3)
        df1.loc[5, "cholesterol"] = 0
        merged = merge_datasets(df1, _make_df(100, seed=4), _make_df(100, seed=5))
        self.assertTrue((merged["cholesterol"] > 0).all())

class TestTrainAndEvaluate(unittest.TestCase):

     #create fake data and train model once for all tests
    def setUp(self):
        df = _make_df(400, seed=20)
        (self.X_train_s, self.X_test_s,
         self.X_train, self.X_test,
         self.y_train,self.y_test, _) = split_and_scale(df)
        self.model = train_model(self.X_train_s, self.y_train)

    #check model has coefficients after training
    def test_model_fitted(self):
        self.assertTrue(hasattr(self.model, "coef_"), "Model should have coef_ attribute after fitting")

     #check all metric keys exist and are valid probabilities between 0 and 1
    def test_evaluate_returns_metrics(self):
        metrics = evaluate_model(self.model, self.X_test_s, self.y_test)
        for key in ["accuracy", "precision", "recall", "f1", "auc"]:
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key],1.0)
            
    #check VIF for each feature and all values are positive 
    def test_vif_shape(self):
        vif = check_multicollinearity(self.X_train_s)
        self.assertEqual(len(vif), len(FEATURE_COLS))
        self.assertTrue((vif["VIF"] > 0).all())

if __name__ == "__main__":
    unittest.main(verbosity=2)