"""Tabular foundation-model adapters with the same factory signature as baselines/common.py:
    make_*(num_cols, cat_cols, seed) -> estimator with fit(X: DataFrame, y) / predict_proba(X)

All machine-specific settings come from environment variables (optionally a `.env` file at the repo root,
see `.env.example`), so no machine-specific values live in the code:

  FM_DEVICE                 auto | cpu | cuda | cuda:0          (default auto)
  FM_PRECISION              auto | fp32 | bf16 | fp16           (default auto = package default: TabPFN "auto",
                            Kumo fp16 autocast on CUDA). fp16/bf16 = mixed-precision torch.autocast in that
                            dtype (weights stay fp32) for both models; fp32 = full precision, no autocast.
  FM_MAX_CONTEXT            max training rows given to an FM as context; larger train sets are stratified-
                            subsampled (0 = no limit, default 0)
  TABPFN_MODEL_PATH         local checkpoint (.safetensors/.ckpt); empty -> download from Hugging Face
                            (Prior-Labs/tabpfn_3_5 at a pinned commit, public; TABPFN-3.5 License v1.0, non-commercial)
  TABPFN_MODEL_VERSION      v3.5 | v3.5-fast -> Hugging Face file; v3 | v2.5 ... -> the tabpfn package's own download
                            (needs TABPFN_TOKEN)                 (default v3.5; used when no model path)
  TABPFN_HF_REPO, TABPFN_HF_REVISION, TABPFN_HF_FILE   override the Hugging Face repo / commit / file
  TABPFN_MODEL_CACHE_DIR    where the Hugging Face file is stored (default <HF_HOME>/tabpfn/<revision>)
  TABPFN_N_ESTIMATORS       default 8
  TABPFN_TOKEN, TABPFN_ALLOW_CPU_LARGE_DATASET   read by the tabpfn package itself
  TABICL_MODEL_PATH         local checkpoint; empty -> download from Hugging Face
  TABICL_CHECKPOINT_VERSION default tabicl-classifier-v2-20260212.ckpt
  TABICL_N_ESTIMATORS       default 8
  TABICL_BATCH_SIZE         ensemble members processed together (default 8; lower to save memory)
  FM_CPU_THREADS            torch threads when running on CPU (default 4)
  KUMO_SIZE                 small | medium | large               (default large)
  KUMO_CHECKPOINT_DIR       optional dir containing <size>/classifier.pt (or classifier.pt directly);
                            empty -> HF cache / download
                            (nvidia/Kumo-Tabular rev v1.0.1; set HF_HUB_CACHE / HF_HUB_OFFLINE=1 for offline use)
  KUMO_N_ESTIMATORS         default 8

Without local paths all three models download their weights from Hugging Face on first use: TabICL and Kumo into the
HF hub cache (HF_HOME / HF_HUB_CACHE), TabPFN into TABPFN_MODEL_CACHE_DIR; HF_HUB_OFFLINE=1 then runs offline.
"""
import contextlib
import functools
import os
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:  # optional; plain environment variables work without it
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_ROOT, ".env"), override=False)
except ImportError:
    pass


def env(name, default=None, cast=str):
    v = os.environ.get(name, "")
    return cast(v) if v != "" else default


def device():
    d = env("FM_DEVICE", "auto")
    if d == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


TABPFN_HF_REPO = "Prior-Labs/tabpfn_3_5"
TABPFN_HF_REVISION = "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e"  # commit of 2026-09-09; sha256 of the v3.5 file: ece4d67e...0be3
TABPFN_HF_FILES = {"v3.5": "tabpfn-v3.5-20260909.safetensors",  # the checkpoint behind every reported TabPFN result
                   "v3.5-fast": "tabpfn-v3.5-fast-20260909.safetensors"}


@functools.lru_cache(maxsize=None)
def tabpfn_checkpoint():
    """Path of the TabPFN checkpoint: TABPFN_MODEL_PATH if set, else the Hugging Face file for TABPFN_MODEL_VERSION,
    downloaded once into TABPFN_MODEL_CACHE_DIR (default <HF_HOME>/tabpfn/<revision>). None for versions without a
    Hugging Face file (tabpfn's own download). The file goes to a plain directory, not the HF hub cache, because the hub
    cache stores it as a symlink to an extension-less blob and tabpfn resolves the link and picks its loader by suffix."""
    path = env("TABPFN_MODEL_PATH")
    if path:
        return path
    file = env("TABPFN_HF_FILE") or TABPFN_HF_FILES.get(env("TABPFN_MODEL_VERSION", "v3.5"))
    if not file:
        return None
    from huggingface_hub import constants, hf_hub_download
    rev = env("TABPFN_HF_REVISION", TABPFN_HF_REVISION)
    return hf_hub_download(env("TABPFN_HF_REPO", TABPFN_HF_REPO), file, revision=rev,
                           local_dir=env("TABPFN_MODEL_CACHE_DIR") or os.path.join(constants.HF_HOME, "tabpfn", rev))


def precision():
    p = env("FM_PRECISION", "auto")
    if p not in ("auto", "fp32", "bf16", "fp16"):
        raise ValueError(f"FM_PRECISION must be auto|fp32|bf16|fp16, got {p!r}")
    return p


class _FMBase(ClassifierMixin, BaseEstimator):
    """Shared preprocessing: numerics as float32 (NaN kept), categoricals ordinal-encoded."""

    def __init__(self, num_cols, cat_cols, seed=0):
        self.num_cols, self.cat_cols, self.seed = num_cols, cat_cols, seed

    def _encode(self, X, fit=False):
        num = X[self.num_cols].to_numpy(dtype="float32")
        if not self.cat_cols:
            return num
        if fit:
            self.enc_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                       encoded_missing_value=np.nan)
            self.enc_.fit(X[self.cat_cols].astype(str))
        cat = self.enc_.transform(X[self.cat_cols].astype(str)).astype("float32")
        return np.hstack([num, cat])

    @property
    def cat_idx_(self):
        return list(range(len(self.num_cols), len(self.num_cols) + len(self.cat_cols)))

    def _context(self, X, y):
        y = np.asarray(y)
        cap = env("FM_MAX_CONTEXT", 0, int)
        if cap and len(y) > cap:
            idx, _ = train_test_split(np.arange(len(y)), train_size=cap, stratify=y, random_state=self.seed)
            X, y = X.iloc[np.sort(idx)], y[np.sort(idx)]
        return X, y

    def fit(self, X, y):
        X, y = self._context(X, y)
        self.classes_ = np.unique(y)
        t0 = time.time()
        self._fit(X, y)
        self.fit_seconds_ = time.time() - t0
        return self

    def predict_proba(self, X):
        p = np.asarray(self._predict_proba(X), dtype="float64")
        return p if p.ndim == 2 else np.column_stack([1 - p, p])


class TabPFNWrap(_FMBase):
    @contextlib.contextmanager
    def _amp(self):
        """TabPFN's autocast uses torch's default autocast dtype (fp16 on CUDA); bf16 swaps that default.
        (Forcing inference_precision=torch.bfloat16 crashes in tabpfn 9.2's fingerprint preprocessing.)"""
        import torch
        dev = torch.device(device()).type
        old = torch.get_autocast_dtype(dev)
        if precision() == "bf16":
            torch.set_autocast_dtype(dev, torch.bfloat16)
        try:
            yield
        finally:
            torch.set_autocast_dtype(dev, old)

    def _fit(self, X, y):
        import torch
        from tabpfn import TabPFNClassifier
        prec = {"auto": "auto", "fp32": torch.float32, "bf16": "autocast", "fp16": "autocast"}[precision()]
        kw = dict(n_estimators=env("TABPFN_N_ESTIMATORS", 8, int), device=device(), random_state=self.seed,
                  categorical_features_indices=self.cat_idx_ or None, ignore_pretraining_limits=True,
                  inference_precision=prec)
        path = tabpfn_checkpoint()
        if path:
            self.m_ = TabPFNClassifier(model_path=path, **kw)
        else:
            from tabpfn.constants import ModelVersion
            self.m_ = TabPFNClassifier.create_default_for_version(ModelVersion(env("TABPFN_MODEL_VERSION", "v3.5")), **kw)
        with self._amp():
            self.m_.fit(self._encode(X, fit=True), y)

    def _predict_proba(self, X):
        with self._amp():
            return self.m_.predict_proba(self._encode(X))


class TabICLWrap(_FMBase):
    def _fit(self, X, y):
        from tabicl import TabICLClassifier
        dev = device()
        self.m_ = TabICLClassifier(
            n_estimators=env("TABICL_N_ESTIMATORS", 8, int), batch_size=env("TABICL_BATCH_SIZE", 8, int), model_path=env("TABICL_MODEL_PATH"),
            checkpoint_version=env("TABICL_CHECKPOINT_VERSION", "tabicl-classifier-v2-20260212.ckpt"),
            device=dev, random_state=self.seed, n_jobs=env("FM_CPU_THREADS", 4, int) if dev == "cpu" else None)
        self.m_.fit(self._encode(X, fit=True), y)

    def _predict_proba(self, X):
        return self.m_.predict_proba(self._encode(X))


class KumoTabularWrap(_FMBase):
    """NVIDIA Kumo-Tabular via structured-data-models (sdm). Categoricals passed natively as strings."""

    def _table(self, X, y=None):
        import sdm
        df = X[self.num_cols + self.cat_cols].copy()
        df[self.num_cols] = df[self.num_cols].astype("float32")
        for c in self.cat_cols:
            df[c] = df[c].astype(str)
        stypes = {c: "numerical" for c in self.num_cols} | {c: "categorical" for c in self.cat_cols}
        if y is not None:
            df["target"] = np.asarray(y).astype(str)
            stypes["target"] = "categorical"
        return sdm.TableTensor.from_pandas(df=df.reset_index(drop=True), stypes=stypes, device=self.dev_)

    def _amp(self):
        import torch
        p = precision()
        dtype = {"auto": torch.float16, "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[p]
        on = p != "fp32" and (p != "auto" or self.dev_.type == "cuda")
        return torch.amp.autocast(self.dev_.type, dtype=dtype, enabled=on)

    def _fit(self, X, y):
        import torch
        from sdm.models import KumoTabular
        self.dev_ = torch.device(device())
        size = env("KUMO_SIZE", "large")
        ckdir = env("KUMO_CHECKPOINT_DIR")
        self.m_ = KumoTabular(task="classification", size=size, pretrained=not ckdir, device=self.dev_)
        if ckdir:
            ck = os.path.join(ckdir, size, "classifier.pt")
            ck = ck if os.path.exists(ck) else os.path.join(ckdir, "classifier.pt")
            sd = torch.load(ck, map_location=self.dev_, weights_only=True)
            next(iter(self.m_.models.values())).load_state_dict(sd, assign=True)
        t = self._table(X, y)
        with self._amp():
            # seeded RNG for the ensemble sampling (otherwise Kumo draws from torch's global RNG)
            self.m_.fit(x=t.drop_columns("target"), y=t[:, "target"], num_estimators=env("KUMO_N_ESTIMATORS", 8, int),
                        generator=torch.Generator(self.dev_).manual_seed(int(self.seed)))

    def _predict_proba(self, X):
        with self._amp():
            out = self.m_.predict(self._table(X)).to_pandas()
        # output columns are per-class probabilities; pick the column for class "1"
        cols = [c for c in out.columns if str(c).endswith("1")]
        if out.shape[1] == 2 and not cols:
            return out.to_numpy(dtype="float64")
        return out[cols[0]].to_numpy(dtype="float64") if cols else out.iloc[:, -1].to_numpy(dtype="float64")


def make_tabpfn(num, cat, seed):
    return TabPFNWrap(num, cat, seed)


def make_tabicl(num, cat, seed):
    return TabICLWrap(num, cat, seed)


def make_kumo(num, cat, seed):
    return KumoTabularWrap(num, cat, seed)


FM_MODELS = {"tabpfn": make_tabpfn, "tabicl": make_tabicl, "kumo_tabular": make_kumo}
