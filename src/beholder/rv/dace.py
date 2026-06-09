"""DACE multi-instrument RV + activity loader.

DACE (https://dace.unige.ch) hosts pipeline-reduced RVs and activity indices
from HARPS, ESPRESSO, HIRES, APF and others. ``Spectroscopy.get_timeseries``
returns a triply-nested dict ``instrument -> drs_id -> mode -> arrays``.

This module flattens that into a tidy long-format DataFrame and applies an
allow-list to drop modes that are not RV-quality. The defaults below were
curated from inspection of the Tau Ceti record:

  - HARPS03 / EGGS:                fast-readout, std ~15 km/s.
  - ESPRESSO18 / SINGLEHR11, …:    early commissioning, std >> precision.
  - ESPRESSO19 / SINGLEHR11:       same pipeline issue (std ~280 m/s).
  - HAMILTON, CORAVEL:             pre-2000, ≥ 2 m/s precision.
  - HIRES (2017 release):          superseded by HIRES-POST04 (Rosenthal+ 2021).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Per-instrument allow-list. Value: set of allowed mode names, or None to
# allow all modes for that instrument.
DEFAULT_ALLOW: dict[str, set[str] | None] = {
    "HARPS03": {"HARPS"},
    "HARPS15": {"HARPS"},
    "HIRES-POST04": None,
    "HIRES-PRE04": None,
    "ESPRESSO19": {"SINGLEHR21"},
    "APF": None,
}

# Fields kept from each DACE record (passed through unchanged where present).
DEFAULT_FIELDS: tuple[str, ...] = (
    "rjd",
    "rv",
    "rv_err",
    "ccf_fwhm",
    "ccf_fwhm_err",
    "ccf_bispan",
    "ccf_bispan_err",
    "ccf_contrast",
    "ccf_contrast_err",
    "spectro_halpha",
    "spectro_halpha_err",
    "spectro_smw",
    "spectro_smw_err",
    "spectro_rhk",
    "spectro_rhk_err",
    "spectro_na",
    "spectro_na_err",
    "drs_qc",
)


def _is_allowed(inst: str, mode: str, allow: dict[str, set[str] | None]) -> bool:
    if inst not in allow:
        return False
    modes = allow[inst]
    if modes is None:
        return True
    return mode in modes


def load_dace_timeseries(
    target: str,
    allow: dict[str, set[str] | None] | None = None,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
) -> pd.DataFrame:
    """Pull DACE timeseries for `target` and return a tidy DataFrame.

    Columns: ``instrument``, ``mode``, ``drs_id``, ``time_rjd``, ``rv``,
    ``rv_err``, plus every member of `fields` that is present on the
    corresponding DACE record.
    """
    from dace_query.spectroscopy import Spectroscopy

    allow = allow if allow is not None else DEFAULT_ALLOW
    raw = Spectroscopy.get_timeseries(target=target, sorted_by_instrument=True)
    rows: list[pd.DataFrame] = []
    for inst_key, drs_dict in raw.items():
        inst = str(inst_key)
        for drs_id, mode_dict in drs_dict.items():
            for mode_key, arrays in mode_dict.items():
                mode = str(mode_key)
                if not _is_allowed(inst, mode, allow):
                    continue
                rjd = arrays.get("rjd")
                rv = arrays.get("rv")
                if rjd is None or rv is None or len(rjd) == 0 or len(rv) == 0:
                    continue
                frame: dict[str, np.ndarray] = {}
                for f in fields:
                    if f in arrays and arrays[f] is not None:
                        frame[f] = np.asarray(arrays[f])
                df = pd.DataFrame(frame).rename(columns={"rjd": "time_rjd"})
                df["instrument"] = inst
                df["mode"] = mode
                df["drs_id"] = str(drs_id)
                rows.append(df)
    if not rows:
        raise RuntimeError(f"No DACE timeseries found for target={target!r}.")
    return pd.concat(rows, ignore_index=True)


def filter_outliers(df: pd.DataFrame, sigma: float = 5.0) -> pd.DataFrame:
    """Sigma-clip RV outliers per (instrument, mode) using a robust MAD scale."""
    keep = pd.Series(False, index=df.index)
    for (_inst, _mode), sub in df.groupby(["instrument", "mode"]):
        rv = sub["rv"].to_numpy()
        med = np.nanmedian(rv)
        mad = np.nanmedian(np.abs(rv - med))
        scale = 1.4826 * mad if mad > 0 else float(np.nanstd(rv))
        ok = np.abs(rv - med) < sigma * scale if scale > 0 else np.ones_like(rv, bool)
        keep.loc[sub.index] = ok
    return df.loc[keep].reset_index(drop=True)


def center_per_instrument(df: pd.DataFrame, value_col: str = "rv") -> pd.DataFrame:
    """Subtract the per-(instrument, mode) median from `value_col`.

    This is the simplest form of zero-point handling — adequate for
    periodograms and exploratory plotting. Joint Bayesian fits should fit
    the offsets as free parameters instead.
    """
    out = df.copy()
    grp = out.groupby(["instrument", "mode"])[value_col]
    out[f"{value_col}_centered"] = out[value_col] - grp.transform("median")
    return out
