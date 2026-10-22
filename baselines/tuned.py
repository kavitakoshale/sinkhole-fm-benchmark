"""Tuned classical baselines: Optuna search with an inner *spatial* group K-fold, TabArena-style spaces.

Same factory signature as common.py: make_<model>_tuned(num, cat, seed) -> estimator with fit(X, y, groups=None) and
predict_proba(X). `groups` are spatial block ids of the training rows (Florida: 25 km `block_id`; Kentucky: ~5 km lon/lat
cells); runners pass them when `est.needs_groups` is True. Without groups the inner CV falls back to a stratified K-fold
and `est.inner_cv_` says so.

Protocol
  * inner CV   : TUNE_INNER_FOLDS (4) GroupKFold over the spatial blocks of the outer training set
  * search     : Optuna TPE, TUNE_TRIALS (100) trials, median pruning after the first inner fold, objective = mean inner
                 TUNE_METRIC (roc_auc | log_loss)
  * GBDTs      : up to TUNE_MAX_ROUNDS (3000) rounds, early stopping (100 rounds) on the inner validation fold; the refit
                 uses the mean best iteration over the inner folds
  * SVM        : TUNE_TRIALS_SVM (40) trials on a stratified subsample of TUNE_SVM_MAX_ROWS (6000) rows (O(n^2) kernel)
  * variants   : "<m>_tuned"      refit of the best configuration on the full outer-train set
                 "<m>_tuned_bag"  mean of the inner-fold models of the best trial (TabArena "tuned + ensembled")
  * cache      : the search result (best params / iterations / value) is cached per (model, data, budget) under
                 TUNE_CACHE_DIR (results/tuning_cache) so the two variants share one search and reruns resume.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from common import N_JOBS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ES_ROUNDS = 100


def _env_int(k, d):
    return int(os.environ.get(k, d))


def budget(base):
    trials = _env_int("TUNE_TRIALS_SVM", 40) if base == "svm_rbf" else _env_int("TUNE_TRIALS", 100)
    return {"n_trials": trials, "inner_folds": _env_int("TUNE_INNER_FOLDS", 4), "metric": os.environ.get("TUNE_METRIC", "roc_auc"),
            "max_rounds": _env_int("TUNE_MAX_ROUNDS", 3000), "svm_max_rows": _env_int("TUNE_SVM_MAX_ROWS", 6000)}


# ----------------------------------------------------------------------------------------------- search spaces (TabArena-like)
def suggest(base, t):
    if base == "xgboost":
        return {"learning_rate": t.suggest_float("learning_rate", 5e-3, 0.1, log=True), "max_depth": t.suggest_int("max_depth", 4, 10),
                "min_child_weight": t.suggest_float("min_child_weight", 1e-3, 5, log=True), "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bylevel": t.suggest_float("colsample_bylevel", 0.6, 1.0), "colsample_bynode": t.suggest_float("colsample_bynode", 0.6, 1.0),
                "reg_alpha": t.suggest_float("reg_alpha", 1e-4, 5, log=True), "reg_lambda": t.suggest_float("reg_lambda", 1e-4, 5, log=True)}
    if base == "lightgbm":
        return {"learning_rate": t.suggest_float("learning_rate", 5e-3, 0.1, log=True), "num_leaves": t.suggest_int("num_leaves", 16, 255, log=True),
                "min_child_samples": t.suggest_int("min_child_samples", 2, 60, log=True), "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
                "subsample": t.suggest_float("subsample", 0.6, 1.0), "reg_lambda": t.suggest_float("reg_lambda", 1e-4, 5, log=True),
                "extra_trees": t.suggest_categorical("extra_trees", [False, True])}
    if base == "catboost":
        return {"learning_rate": t.suggest_float("learning_rate", 5e-3, 0.1, log=True), "depth": t.suggest_int("depth", 4, 8),
                "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 5, log=True), "max_ctr_complexity": t.suggest_int("max_ctr_complexity", 1, 5),
                "one_hot_max_size": t.suggest_categorical("one_hot_max_size", [2, 3, 5, 10]),
                "grow_policy": t.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"])}
    if base == "random_forest":
        return {"max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.75, 1.0]),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20, log=True),
                "max_samples": t.suggest_float("max_samples", 0.5, 1.0)}
    if base == "logreg":
        return {"C": t.suggest_float("C", 1e-4, 1e2, log=True), "l1_ratio": t.suggest_categorical("l1_ratio", [0.0, 0.5, 1.0])}
    if base == "svm_rbf":
        return {"C": t.suggest_float("C", 1e-2, 1e3, log=True), "gamma": t.suggest_float("gamma", 1e-4, 10, log=True)}
    raise KeyError(base)


GBDT = {"xgboost", "lightgbm", "catboost"}


# ----------------------------------------------------------------------------------------------- encoders
class _Ordinal:
    def __init__(self, num, cat):
        self.num, self.cat = num, cat

    def fit(self, X):
        self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
        if self.cat:
            self.enc.fit(X[self.cat].astype(str))
        return self

    def __call__(self, X):
        out = X[self.num].astype("float32").copy()
        if self.cat:
            out[self.cat] = self.enc.transform(X[self.cat].astype(str)).astype("float32")
        return out


def _dense(num, cat, indicator=False):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=indicator)), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), cat)])


# ----------------------------------------------------------------------------------------------- one configured fit
def fit_one(base, params, num, cat, seed, Xtr, ytr, Xva=None, yva=None, n_rounds=None, max_rounds=3000):
    """Fit `base` with `params`. GBDTs early-stop on (Xva, yva) when given, else run `n_rounds`.
    Returns (predict_fn, best_iteration or None)."""
    if base in GBDT:
        rounds = int(n_rounds) if n_rounds else max_rounds
        es = Xva is not None
        if base == "catboost":
            from catboost import CatBoostClassifier
            cols = num + cat
            tx = lambda X: X[cols].assign(**{c: X[c].astype(str) for c in cat})  # noqa: E731
            m = CatBoostClassifier(iterations=rounds, random_seed=seed, verbose=0, thread_count=N_JOBS, cat_features=cat,
                                   allow_writing_files=False, boosting_type="Plain",
                                   **({"early_stopping_rounds": ES_ROUNDS, "use_best_model": True} if es else {}), **params)
            m.fit(tx(Xtr), ytr, eval_set=(tx(Xva), yva) if es else None)
            best = m.get_best_iteration() + 1 if es else None
            return (lambda X: m.predict_proba(tx(X))[:, 1]), best
        enc = _Ordinal(num, cat).fit(Xtr)
        if base == "xgboost":
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=rounds, tree_method="hist", n_jobs=N_JOBS, random_state=seed, eval_metric="logloss",
                              early_stopping_rounds=ES_ROUNDS if es else None, **params)
            m.fit(enc(Xtr), ytr, eval_set=[(enc(Xva), yva)] if es else None, verbose=False)
            best = m.best_iteration + 1 if es else None
        else:
            import lightgbm as lgb
            m = lgb.LGBMClassifier(n_estimators=rounds, subsample_freq=1, random_state=seed, n_jobs=N_JOBS, verbose=-1, **params)
            m.fit(enc(Xtr), ytr, **({"eval_X": enc(Xva), "eval_y": yva, "callbacks": [lgb.early_stopping(ES_ROUNDS, verbose=False)]} if es else {}))
            best = m.best_iteration_ if es else None
        return (lambda X: m.predict_proba(enc(X))[:, 1]), best
    if base == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        enc = _Ordinal(num, cat).fit(Xtr)
        m = RandomForestClassifier(n_estimators=300, n_jobs=N_JOBS, random_state=seed, **params).fit(enc(Xtr), ytr)
        return (lambda X: m.predict_proba(enc(X))[:, 1]), None
    if base == "logreg":
        from sklearn.linear_model import LogisticRegression
        m = make_pipeline(_dense(num, cat, indicator=True), LogisticRegression(solver="saga", max_iter=5000, **params)).fit(Xtr, ytr)
        return (lambda X: m.predict_proba(X)[:, 1]), None
    if base == "svm_rbf":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC
        svc = CalibratedClassifierCV(SVC(kernel="rbf", cache_size=2000, random_state=seed, **params), method="sigmoid", cv=5, ensemble=False)
        m = make_pipeline(_dense(num, cat), svc).fit(Xtr, ytr)
        return (lambda X: m.predict_proba(X)[:, 1]), None
    raise KeyError(base)


def _score(metric, y, p):
    return roc_auc_score(y, p) if metric == "roc_auc" else -log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))


# ----------------------------------------------------------------------------------------------- the estimator
class Tuned(ClassifierMixin, BaseEstimator):
    needs_groups = True

    def __init__(self, base, num_cols, cat_cols, seed=0, bag=False):
        self.base, self.num_cols, self.cat_cols, self.seed, self.bag = base, num_cols, cat_cols, seed, bag

    # -- inner folds
    def _inner(self, X, y, groups, k):
        if groups is not None and pd.Series(groups).nunique() >= k:
            self.inner_cv_ = f"GroupKFold({k}) over spatial blocks"
            g = pd.factorize(np.asarray(groups))[0]
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(g.max() + 1)  # shuffle block order so folds are not spatially ordered
            return list(GroupKFold(n_splits=k).split(X, y, perm[g]))
        self.inner_cv_ = f"StratifiedKFold({k}) (no groups given)"
        return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed).split(X, y))

    def _cache_key(self, X, y, groups, b):
        h = hashlib.sha1()
        h.update(json.dumps([self.base, self.seed, b, list(self.num_cols), list(self.cat_cols)], sort_keys=True, default=str).encode())
        h.update(np.ascontiguousarray(X.index.to_numpy()).tobytes())
        h.update(np.ascontiguousarray(np.asarray(y)).tobytes())
        if groups is not None:
            h.update(pd.util.hash_pandas_object(pd.Series(np.asarray(groups)), index=False).to_numpy().tobytes())
        return h.hexdigest()[:20]

    def _search(self, X, y, folds, b):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        Xs, ys, fs = X, y, folds
        if self.base == "svm_rbf" and len(X) > b["svm_max_rows"]:  # tune on a subsample, refit on everything
            idx, _ = train_test_split(np.arange(len(X)), train_size=b["svm_max_rows"], stratify=y, random_state=self.seed)
            idx = np.sort(idx)
            pos = {i: j for j, i in enumerate(idx)}
            Xs, ys = X.iloc[idx], y[idx]
            fs = [(np.array([pos[i] for i in tr if i in pos]), np.array([pos[i] for i in va if i in pos])) for tr, va in folds]

        def objective(t):
            params = suggest(self.base, t)
            scores, iters = [], []
            for i, (tr, va) in enumerate(fs):
                if len(np.unique(ys[tr])) < 2 or len(np.unique(ys[va])) < 2:  # tiny n + spatial groups: single-class inner fold
                    iters.append(None)
                    continue
                pred, best = fit_one(self.base, params, self.num_cols, self.cat_cols, self.seed, Xs.iloc[tr], ys[tr], Xs.iloc[va], ys[va],
                                     max_rounds=b["max_rounds"])
                scores.append(_score(b["metric"], ys[va], pred(Xs.iloc[va])))
                iters.append(best)
                t.report(float(np.mean(scores)), i)
                if t.should_prune():
                    raise optuna.TrialPruned()
            t.set_user_attr("best_iters", iters)
            return float(np.mean(scores)) if scores else 0.5

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed, multivariate=True),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0))
        study.optimize(objective, n_trials=b["n_trials"], n_jobs=_env_int("TUNE_N_JOBS", 1))
        bt = study.best_trial
        return {"best_params": bt.params, "best_value": bt.value, "best_iters": bt.user_attrs.get("best_iters"),
                "n_trials": len(study.trials), "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)}

    def fit(self, X, y, groups=None):
        t0 = time.time()
        y = np.asarray(y)
        b = budget(self.base)
        folds = self._inner(X, y, groups, b["inner_folds"])
        cache_dir = os.environ.get("TUNE_CACHE_DIR", f"{ROOT}/results/tuning_cache")
        os.makedirs(cache_dir, exist_ok=True)
        path = f"{cache_dir}/{self.base}_{self._cache_key(X, y, groups, b)}.json"
        if os.path.exists(path):
            res = json.load(open(path))
        else:
            res = self._search(X, y, folds, b)
            res.update(search_s=time.time() - t0, inner_cv=self.inner_cv_, budget=b)
            json.dump(res, open(path, "w"), indent=1, default=str)
        self.search_ = res
        p = res["best_params"]
        iters = [i for i in (res.get("best_iters") or []) if i]
        if self.bag:
            self.predict_fns_ = [fit_one(self.base, p, self.num_cols, self.cat_cols, self.seed, X.iloc[tr], y[tr],
                                         n_rounds=iters[i] if iters else None, max_rounds=b["max_rounds"])[0]
                                 for i, (tr, _) in enumerate(folds)]
        else:
            self.predict_fns_ = [fit_one(self.base, p, self.num_cols, self.cat_cols, self.seed, X, y,
                                         n_rounds=int(np.mean(iters)) if iters else None, max_rounds=b["max_rounds"])[0]]
        self.classes_ = np.array([0, 1])
        self.fit_s_ = time.time() - t0
        return self

    def predict_proba(self, X):
        p = np.mean([f(X) for f in self.predict_fns_], axis=0)
        return np.column_stack([1 - p, p])


def _factory(base, bag):
    return lambda num, cat, seed: Tuned(base, num, cat, seed, bag)


TUNED_MODELS = {f"{b}_tuned": _factory(b, False) for b in ["xgboost", "lightgbm", "catboost", "random_forest", "logreg", "svm_rbf"]}
TUNED_MODELS.update({f"{b}_tuned_bag": _factory(b, True) for b in GBDT})
