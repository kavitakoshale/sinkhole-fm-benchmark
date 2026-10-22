"""Shared model zoo + evaluation utilities for tabular geohazard baselines.

Every model is a factory `f(num_cols, cat_cols, seed) -> estimator` exposing fit(X, y) / predict_proba(X),
where X is a pandas DataFrame with numeric columns (NaN allowed) and categorical columns as strings.
Foundation models (TabPFN, TabICL, ...) will be added to MODELS with the same signature so they are
evaluated on exactly the same folds and metrics.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss,
                             matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

N_JOBS = 4  # see run_*.py: more threads per process oversubscribe the cores


class OrdinalWrap(ClassifierMixin, BaseEstimator):
    """Ordinal-encode categoricals (unknown -> -1), pass numerics through with NaN; then fit `est`."""

    def __init__(self, est, num_cols, cat_cols):
        self.est, self.num_cols, self.cat_cols = est, num_cols, cat_cols

    def _tx(self, X):
        out = X[self.num_cols].astype("float32").copy()
        if self.cat_cols:
            out[self.cat_cols] = self.enc_.transform(X[self.cat_cols].astype(str)).astype("float32")
        return out

    def fit(self, X, y):
        self.enc_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
        if self.cat_cols:
            self.enc_.fit(X[self.cat_cols].astype(str))
        self.est_ = clone(self.est).fit(self._tx(X), y)
        self.classes_ = self.est_.classes_
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(self._tx(X))


def make_lr(num, cat, seed):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat),
    ])
    return make_pipeline(pre, LogisticRegression(C=1.0, max_iter=5000))


def make_rf(num, cat, seed):
    return OrdinalWrap(RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                              n_jobs=N_JOBS, random_state=seed), num, cat)


def make_xgb(num, cat, seed):
    from xgboost import XGBClassifier
    return OrdinalWrap(XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
                                     colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0, tree_method="hist",
                                     n_jobs=N_JOBS, random_state=seed, eval_metric="logloss"), num, cat)


def make_xgb_default(num, cat, seed):
    from xgboost import XGBClassifier
    return OrdinalWrap(XGBClassifier(tree_method="hist", n_jobs=N_JOBS, random_state=seed), num, cat)


def make_lgbm(num, cat, seed):
    from lightgbm import LGBMClassifier
    return OrdinalWrap(LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=N_JOBS,
                                      verbose=-1), num, cat)


class CatBoostWrap(ClassifierMixin, BaseEstimator):
    def __init__(self, num_cols, cat_cols, seed=0):
        self.num_cols, self.cat_cols, self.seed = num_cols, cat_cols, seed

    def _tx(self, X):
        out = X[self.num_cols + self.cat_cols].copy()
        out[self.cat_cols] = out[self.cat_cols].astype(str)
        return out

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        self.m_ = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, random_seed=self.seed,
                                     verbose=0, thread_count=N_JOBS, cat_features=self.cat_cols,
                                     allow_writing_files=False)
        self.m_.fit(self._tx(X), y)
        self.classes_ = self.m_.classes_
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(self._tx(X))


def make_cat(num, cat, seed):
    return CatBoostWrap(num, cat, seed)


def _dense_pre(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), cat),
    ])


def make_mlp10(num, cat, seed):
    """Zhu et al. (2020) best model: one hidden layer, 10 nodes."""
    from sklearn.neural_network import MLPClassifier
    return make_pipeline(_dense_pre(num, cat), MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000,
                                                             early_stopping=True, random_state=seed))


def make_svm_poly2(num, cat, seed):
    """Zhu et al. (2020): SVM with quadratic polynomial kernel (Platt-scaled probabilities)."""
    from sklearn.svm import SVC
    return make_pipeline(_dense_pre(num, cat), SVC(kernel="poly", degree=2, probability=True, random_state=seed))


def make_nb(num, cat, seed):
    from sklearn.naive_bayes import GaussianNB
    return make_pipeline(_dense_pre(num, cat), GaussianNB())


MODELS = {
    "logreg": make_lr,
    "random_forest": make_rf,
    "xgboost_default": make_xgb_default,
    "xgboost": make_xgb,
    "lightgbm": make_lgbm,
    "catboost": make_cat,
}

# models from the replicated papers; used where the paper used them (Kentucky: Zhu et al. 2020)
PAPER_MODELS = {"naive_bayes": make_nb, "mlp10": make_mlp10, "svm_poly2": make_svm_poly2}


def metrics(y, p, thr=0.5):
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    yhat = (p >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p),
        "accuracy": accuracy_score(y, yhat), "f1": f1_score(y, yhat), "mcc": matthews_corrcoef(y, yhat),
        "n_test": len(y), "pos_rate_test": float(y.mean()),
    }


def score_only_metrics(y, s):
    """For expert maps that give a ranking score but no calibrated probability."""
    m = ~np.isnan(s)
    return {"roc_auc": roc_auc_score(np.asarray(y)[m], s[m]),
            "pr_auc": average_precision_score(np.asarray(y)[m], s[m]),
            "n_test": int(m.sum()), "pos_rate_test": float(np.asarray(y)[m].mean())}


def summarize(long_df, by):
    num = ["roc_auc", "pr_auc", "brier", "log_loss", "accuracy", "f1", "mcc"]
    num = [c for c in num if c in long_df]
    g = long_df.groupby(by)[num]
    out = g.mean().round(4).astype(str) + " ± " + g.std().fillna(0).round(4).astype(str)
    return out.reset_index()


def as_frame(X):
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
