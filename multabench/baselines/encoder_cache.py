"""Opt-in, run-scoped memoization of the (learner-independent) text/image encoder fits.

MulTaBench fits the E5/DINO encoders (frozen encode *and* LoRA finetune) inside every
learner's ``fit_preprocessor``, but a fitted encoder depends only on the data, the tune
flag, and the train kwargs — never on the learner. An evaluation that runs several
learners over one dataset therefore refits the same encoder repeatedly, the LoRA finetune
being the dominant cost. Enabling this cache around such a run fits each distinct encoder
once and shares it across learners (collapsing e.g. 5 finetunes to 1, and deduping the
frozen encode shared by the text-only and joint-text conditions).

Disabled by default: when off, the wrappers below delegate straight through, so ordinary
single-model runs are byte-for-byte unchanged. Enable explicitly via ``enable_encoder_cache``
or, preferably, the ``encoder_cache`` context manager (which also clears on entry/exit).

Isolation / no val leakage: the cache key hashes the exact columns *and rows* fed in, plus
the model/tune/train/encode config. Learner groups that see different train rows (e.g.
val-split learners get x_train−val while full-train learners get x_train) hash differently
and never share a fit; learners within a group hash identically (splits are seeded) and hit
the cache.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import Any, Dict

import pandas as pd

from multabench.baselines.preprocessing.text_embeddings import fit_text_encoders
from multabench.baselines.preprocessing.image_embeddings import fit_image_encoders
from multabench.baselines.preprocessing.feature_types import detect_image_features

_CACHE: Dict[Any, Any] = {}
_ENABLED: bool = False


def enable_encoder_cache() -> None:
    """Turn the cross-learner encoder cache on (leaves existing entries in place)."""
    global _ENABLED
    _ENABLED = True


def disable_encoder_cache() -> None:
    global _ENABLED
    _ENABLED = False


def clear_encoder_cache() -> None:
    _CACHE.clear()


@contextmanager
def encoder_cache():
    """Enable the run-scoped encoder cache for the duration of the block.

    Clears the cache on entry and exit and restores the previous enabled state, so a run's
    fits never leak into another run.
    """
    global _ENABLED
    prev = _ENABLED
    _ENABLED = True
    clear_encoder_cache()
    try:
        yield
    finally:
        clear_encoder_cache()
        _ENABLED = prev


def _hash_frame(obj) -> str:
    """Stable content hash of a DataFrame/Series (values + index)."""
    return hashlib.sha1(pd.util.hash_pandas_object(obj, index=True).values.tobytes()).hexdigest()


def _kwargs_key(d) -> tuple:
    return tuple(sorted((d or {}).items()))


def cached_fit_text_encoders(**kw):
    """Memoizing wrapper around :func:`fit_text_encoders`; delegates unchanged when disabled."""
    if not _ENABLED:
        return fit_text_encoders(**kw)
    cols = sorted(kw.get("text_features") or set())
    if not cols:
        return fit_text_encoders(**kw)
    tune, y = kw.get("tune_e5", False), kw.get("y")
    key = ("text", kw.get("e5_model_name"), tune, _kwargs_key(kw.get("e5_train_kwargs")),
           _kwargs_key(kw.get("e5_encode_kwargs")), kw.get("is_cls"), kw.get("d_output"),
           kw.get("pca_components"), kw.get("no_pca"), tuple(cols),
           _hash_frame(kw["x"][cols]),
           _hash_frame(y) if (tune and y is not None) else None)
    if key in _CACHE:
        print(f"  ♻️  reuse E5 encoder (tune={tune}) — cache hit")
        return _CACHE[key]
    enc = fit_text_encoders(**kw)
    _CACHE[key] = enc
    return enc


def cached_fit_image_encoders(**kw):
    """Memoizing wrapper around :func:`fit_image_encoders`; delegates unchanged when disabled."""
    if not _ENABLED:
        return fit_image_encoders(**kw)
    x = kw["x"]
    img_cols = sorted(detect_image_features(x=x))
    if not img_cols or kw.get("image_folder") is None:
        return fit_image_encoders(**kw)
    tune, y = kw.get("tune_dino", False), kw.get("y")
    key = ("image", kw.get("dino_model_name"), kw.get("image_folder"), tune,
           _kwargs_key(kw.get("dino_train_kwargs")), kw.get("is_cls"), kw.get("d_output"),
           kw.get("pca_components"), kw.get("no_pca"), tuple(img_cols),
           _hash_frame(x[img_cols]), _hash_frame(y) if (tune and y is not None) else None)
    if key in _CACHE:
        print(f"  ♻️  reuse DINO encoder (tune={tune}) — cache hit")
        return _CACHE[key]
    out = fit_image_encoders(**kw)
    _CACHE[key] = out
    return out
