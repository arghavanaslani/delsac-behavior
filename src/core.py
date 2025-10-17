from __future__ import annotations
import pickle
from typing import Iterable, List, Dict, Optional, Sequence, Tuple, Any
import pandas as pd
import numpy as np
import re
import ast
from pathlib import Path
import scipy.io



def load_data(filepath):
    """
    Load pickled data from a given filepath.

    Parameters
    ----------
    filepath : str
        Path to the pickle (.pkl) file.

    Returns
    -------
    data : object
        The loaded data object from the pickle file.
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def load_eye_data_for_subject(
    base_dir: str | Path,
    sessions: Iterable[str],
    array_key: str = "ain",
    trial_axis: int = 2
) -> Tuple[Dict[str, dict], List[str], int]:
    """
    Convenience wrapper: find files, load them, and return (data, missing_sessions, total_trials).

    Returns
    -------
    mat_dict : dict
        {session_id: mat_dict}
    missing : list of str
        Sessions with no matching .mat file.
    total_trials : int
        Total trials across loaded sessions for `array_key`.
    """
    files, missing = find_session_mat_files(base_dir, sessions)
    mat_dict = load_mat_dict(files)
    total_trials = count_total_trials(mat_dict, array_key=array_key, trial_axis=trial_axis)
    return mat_dict, missing, total_trials


def apply_mapping_for_session(df_sess: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    needed = ["correct","targX","targY","endX_raw","endY_raw","fixBaseX_raw","fixBaseY_raw"]
    good = (df_sess["correct"] == 1) & df_sess[needed].notna().all(axis=1)

    fit_info = {"A": None, "rmse_x": np.nan, "rmse_y": np.nan, "rmse": np.nan}

    if not np.any(good):
        df_sess = df_sess.copy()
        df_sess["targX_raw_pred"] = np.nan
        df_sess["targY_raw_pred"] = np.nan
        df_sess["endErr_raw"]     = np.nan
        return df_sess, fit_info

    Tm = df_sess.loc[good, ["targX","targY"]].to_numpy(float)
    Rm = df_sess.loc[good, ["endX_raw","endY_raw"]].to_numpy(float)
    Bm = df_sess.loc[good, ["fixBaseX_raw","fixBaseY_raw"]].to_numpy(float)
    R_rel = Rm - Bm

    A, *_ = np.linalg.lstsq(Tm, R_rel, rcond=None)  # (2x2)
    fit_info["A"] = A

    # predict for ALL rows in this session (even if some are NaN → result NaN)
    Tall = df_sess[["targX","targY"]].to_numpy(float)
    Ball = df_sess[["fixBaseX_raw","fixBaseY_raw"]].to_numpy(float)
    R_pred_rel = Tall @ A
    R_pred_abs = Ball + R_pred_rel

    df_out = df_sess.copy()
    df_out["targX_raw_pred"] = R_pred_abs[:, 0]
    df_out["targY_raw_pred"] = R_pred_abs[:, 1]

    end_xy = df_sess[["endX_raw","endY_raw"]].to_numpy(float)
    df_out["endErr_raw"] = np.hypot(end_xy[:,0] - R_pred_abs[:,0],
                                    end_xy[:,1] - R_pred_abs[:,1])

    # training RMSE (baseline-relative) for reference
    train_pred_rel = Tm @ A
    rmse_axis = np.sqrt(np.nanmean((train_pred_rel - R_rel)**2, axis=0))
    fit_info["rmse_x"] = float(rmse_axis[0])
    fit_info["rmse_y"] = float(rmse_axis[1])
    fit_info["rmse"]   = float(np.sqrt(np.nanmean((train_pred_rel - R_rel)**2)))

    return df_out, fit_info



STASH_COL = "_df_idx_index_"


def _to_session_id(v) -> str:
    """
    Convert an index-ish value to a plain session_id string (strip extension).
    Handles scalars, 1-elem list/tuple/ndarray.
    """
    if isinstance(v, (list, tuple)):
        s = v[0]
    else:
        try:
            arr = np.asarray(v)
            s = arr.flat[0] if arr.shape else v
        except Exception:
            s = v
    return str(s).split(".")[0]


def prepare_df_idx_for_merge(
    df_idx: pd.DataFrame,
    trial_col: str = "trial",
    session_col: str = "session_id",
) -> pd.DataFrame:
    """
    Make a merge-ready view of df_idx while preserving its original index.

    - Stashes the original index into STASH_COL so we can restore it later.
    - Ensures a clean 'session_id' derived from the original index if missing.
    - Keeps only rows with non-null `trial`, casting to int.
    """
    dfi = df_idx.copy()
    dfi[STASH_COL] = dfi.index

    if session_col not in dfi.columns:
        dfi[session_col] = dfi[STASH_COL].map(_to_session_id)

    if trial_col not in dfi.columns:
        raise KeyError(f"`{trial_col}` column not found in df_idx.")

    dfi = dfi.dropna(subset=[trial_col]).copy()
    dfi[trial_col] = pd.to_numeric(dfi[trial_col], errors="coerce").astype(int)
    dfi[session_col] = dfi[session_col].astype(str)

    return dfi


def prepare_df_trials_for_merge(
    df_trials: pd.DataFrame,
    trial_id_col: str = "trial_id",
    session_col: str = "session_id",
) -> pd.DataFrame:
    """
    Normalize df_trials types for merging.
    """
    dft = df_trials.copy()
    if trial_id_col not in dft.columns:
        raise KeyError(f"`{trial_id_col}` column not found in df_trials.")
    if session_col not in dft.columns:
        raise KeyError(f"`{session_col}` column not found in df_trials.")

    dft[trial_id_col] = pd.to_numeric(dft[trial_id_col], errors="coerce").astype(int)
    dft[session_col]  = dft[session_col].astype(str)
    return dft



def session_idx_table(
    df_idx_r: pd.DataFrame,
    sess_id: str,
    kept_trial_ids: np.ndarray,
) -> pd.DataFrame:
    sub = df_idx_r[(df_idx_r["session_id"] == str(sess_id)) & (df_idx_r["trial"].notna())]
    sub = sub.set_index("trial")
    cols = ["responseTime_idx","responseDone_idx","fixptOff_idx","targX","targY","correct"]
    # reindex to align to kept_trial_ids order (kept_trial_ids are ORIGINAL trial numbers)
    tbl = sub.reindex(kept_trial_ids)[cols]
    tbl = tbl.reset_index().rename(columns={"index":"trial"})
    # normalize dtypes to numeric where applicable
    for c in ["responseTime_idx","responseDone_idx","fixptOff_idx","targX","targY","correct"]:
        tbl[c] = pd.to_numeric(tbl[c], errors="coerce")
    # keep pandas nullable ints for 'correct'
    tbl["correct"] = tbl["correct"].astype("Int64")
    return tbl.reset_index(drop=True)




def _clip_idx_small(i, n_samples: int) -> int:
    """Clamp round(i) into [0, n_samples-1] — banker's round(), same as small code."""
    return int(max(0, min(n_samples - 1, round(float(i)))))

def _endpoint_window(c: int, halfwin: int, n_samples: int) -> Tuple[int, int]:
    """Centered window, inclusive of center sample."""
    s = max(0, c - halfwin)
    e = min(n_samples, c + halfwin + 1)  # inclusive end
    return s, e

def _baseline_window(e: int, win: int, n_samples: int) -> Tuple[int, int]:
    """Immediately before event, exclusive of event sample."""
    s = max(0, e - win)
    return s, min(n_samples, e)          # [s, e)


def compute_features_for_session(
    ain: np.ndarray,
    idx_tbl: pd.DataFrame,
    pre_go_win: int = 25,
    halfwin: int = 10
) -> pd.DataFrame:
    """
    ain: (S, C>=2, T) with ch0=X, ch1=Y
    idx_tbl: rows aligned to trial axis order (0..T-1)
    returns DataFrame with fixBaseX_raw, fixBaseY_raw, endX_raw, endY_raw
            plus targX, targY, correct, response indices
    """
    S, C, T = ain.shape
    assert C >= 2, "AIN must have at least two channels (X=0, Y=1)."

    endX = np.full(T, np.nan, float)
    endY = np.full(T, np.nan, float)
    baseX = np.full(T, np.nan, float)
    baseY = np.full(T, np.nan, float)

    for i in range(T):
        # Endpoint around responseDone
        t_done = idx_tbl.at[i, "responseDone_idx"]
        if pd.notna(t_done):
            c = _clip_idx_small(t_done, S)
            s, e = _endpoint_window(c, halfwin, S)
            x = ain[s:e, 0, i]
            y = ain[s:e, 1, i]
            if x.size:
                endX[i] = float(np.nanmean(x))
                endY[i] = float(np.nanmean(y))

        # Baseline before fixptOff
        t_go = idx_tbl.at[i, "fixptOff_idx"]
        if pd.notna(t_go):
            e2 = _clip_idx_small(t_go, S)
            s2, e2b = _baseline_window(e2, pre_go_win, S)
            bx = ain[s2:e2b, 0, i]
            by = ain[s2:e2b, 1, i]
            if bx.size:
                baseX[i] = float(np.nanmean(bx))
                baseY[i] = float(np.nanmean(by))

    # NOTE: keep 'correct' as pandas nullable Int64; use .to_numpy(float) for others
    out = pd.DataFrame({
        "fixBaseX_raw": baseX,
        "fixBaseY_raw": baseY,
        "endX_raw": endX,
        "endY_raw": endY,
        "targX": idx_tbl["targX"].to_numpy(float),
        "targY": idx_tbl["targY"].to_numpy(float),
        "correct": idx_tbl["correct"].astype("Int64"),  # <- no .to_numpy("Int64")
        "responseTime_idx": idx_tbl["responseTime_idx"].to_numpy(float),
        "responseDone_idx": idx_tbl["responseDone_idx"].to_numpy(float),
        "fixptOff_idx": idx_tbl["fixptOff_idx"].to_numpy(float),
    })
    return out




def build_trials_table_for_subject(
    filtered_mat_data_dict: Dict[str, dict],
    df_idx_r: pd.DataFrame,
    pre_go_win: int = 25,
    halfwin: int = 10,
) -> tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Build df_trials for a subject using the **original working logic**.

    Parameters
    ----------
    filtered_mat_data_dict : {session_id: {'ain': (S,C,T), 'kept_trial_ids': (T,), ...}}
        If 'kept_trial_ids' is missing, we fall back to 1..T (1-based), preserving behavior you saw.
    df_idx_r : DataFrame
        Must contain 'session_id', 'trial', and the timing/target columns used by session_idx_table.

    Returns
    -------
    df_trials : pd.DataFrame
        Concatenated per-trial features + predictions across all sessions.
    fit_summaries : dict
        {session_id: {'A': 2x2, 'rmse_x': float, 'rmse_y': float, 'rmse': float}}
    """
    all_sessions_rows: List[pd.DataFrame] = []
    fit_summaries: Dict[str, dict] = {}

    for sess_id, sess_dict in filtered_mat_data_dict.items():
        ain = sess_dict["ain"]  # (S, C>=2, T_kept)

        # Safe fallback if kept_trial_ids isn't present
        kept_trial_ids = sess_dict.get("kept_trial_ids", None)
        if kept_trial_ids is None:
            T = ain.shape[-1]
            # 1-based, as in your fallback message
            kept_trial_ids = np.arange(1, T + 1, dtype=int)

        # 1) get df rows aligned to kept_trial_ids order
        idx_tbl = session_idx_table(df_idx_r, str(sess_id), np.asarray(kept_trial_ids, dtype=int))

        # 2) compute features like small code
        feats = compute_features_for_session(ain, idx_tbl, pre_go_win=pre_go_win, halfwin=halfwin)

        # 3) attach session and original trial id
        feats.insert(0, "trial_id", np.asarray(kept_trial_ids, dtype=int))
        feats.insert(0, "session_id", str(sess_id))

        # 4) per-session mapping & predictions
        feats_pred, fit_info = apply_mapping_for_session(feats)

        fit_summaries[str(sess_id)] = fit_info
        all_sessions_rows.append(feats_pred)

    df_trials = pd.concat(all_sessions_rows, ignore_index=True) if all_sessions_rows else pd.DataFrame()
    return df_trials, fit_summaries



def merge_trials_with_df_idx(
    df_trials: pd.DataFrame,
    df_idx: pd.DataFrame,
    trial_id_col: str = "trial_id",
    trial_col: str = "trial",
    session_col: str = "session_id",
) -> pd.DataFrame:
    """
    Merge df_trials (per-trial features) into df_idx on (session_id, trial),
    then restore df_idx’s original index on the merged result.
    """
    dfi = prepare_df_idx_for_merge(df_idx, trial_col=trial_col, session_col=session_col)
    dft = prepare_df_trials_for_merge(df_trials, trial_id_col=trial_id_col, session_col=session_col)

    merged = pd.merge(
        dft,
        dfi,
        left_on=[session_col, trial_id_col],
        right_on=[session_col, trial_col],
        how="inner",
        suffixes=("_drop", ""),  # keep df_idx columns clean
        validate="m:m",
    )

    # Restore df_idx's original index
    merged = merged.set_index(STASH_COL)
    merged.index.name = df_idx.index.name

    # Drop right-side join key and any left-side duplicates
    to_drop = [trial_col] + [c for c in merged.columns if c.endswith("_drop")]
    merged = merged.drop(columns=[c for c in to_drop if c in merged.columns])

    return merged





def select_correct_trials(
    data: Dict,
    task_key: str = "delsac",
    required_cols: Sequence[str] = ("correct", "badTrials", "badTimingTrials"),
    drop_cols: Sequence[str] = ("feedbackOff",),
    subject: Optional[str] = None,
    subject_field: str = "subject"
) -> List[Dict[str, pd.DataFrame]]:
    """
    Filter sessions to keep only correct, non-bad, non-bad-timing trials for a given task.

    Parameters
    ----------
    data : dict
        The loaded data object. Expected to contain a "trial" iterable of session dicts.
    task_key : str, default "delsac"
        The key within each session dict that holds the pandas DataFrame for this task.
    required_cols : sequence of str
        Columns that must be present to apply filtering.
    drop_cols : sequence of str
        Columns to drop from each session DataFrame if present (e.g., "feedbackOff").
    subject : str or None
        If provided (e.g., "paula" or "rex"), only sessions matching this subject are processed.
        Matching is done by comparing `str(sess.get(subject_field, "")).lower()`.
        If the field is absent, sessions are not excluded.
    subject_field : str
        The key in each session dict that stores the subject identifier.

    Returns
    -------
    filtered_sessions : list of dict
        Each item is {task_key: DataFrame} for sessions where filtering was possible.

    Notes
    -----
    - Sessions without the specified `task_key` are skipped.
    - Sessions missing any `required_cols` are skipped.
    - DataFrames are copied before filtering to avoid mutating the original data.
    """
    sessions: Iterable[Dict] = data.get("trial", [])
    filtered: List[Dict[str, pd.DataFrame]] = []

    for sess in sessions:
        # Optional subject filtering (if requested and field is present)
        if subject is not None:
            subj_value = str(sess.get(subject_field, "")).lower()
            if subj_value and subj_value != subject.lower():
                continue  # skip non-matching subjects

        if task_key not in sess:
            continue

        df = sess[task_key]
        if not isinstance(df, pd.DataFrame):
            continue

        # Ensure required columns exist
        if not set(required_cols).issubset(df.columns):
            continue

        # Apply filters
        mask = (
            (df["correct"] == 1)
            & (df["badTrials"] == 0)
            & (df["badTimingTrials"] == 0)
        )
        df_filt = df.loc[mask].copy()

        # Drop optional columns if present
        cols_to_drop = [c for c in drop_cols if c in df_filt.columns]
        if cols_to_drop:
            df_filt.drop(columns=cols_to_drop, inplace=True)

        filtered.append({task_key: df_filt})

    return filtered




# Default constants
FS_DEFAULT = 1000.0     # Hz
T_START_DEFAULT = -2.5  # s
N_DEFAULT = 6001        # samples (−2.5 … +3.5 at 1 kHz)
TIME_COLS_DEFAULT = [
    'trialStart', 'fixptOn', 'fixationTime', 'fixptOn2',
    'stimOn', 'stimOff', 'targetOn', 'targetOff',
    'fixptOff', 'responseTime', 'responseDone'
]


#  Core conversions 

def time_to_idx(series: pd.Series,
                fs: float = FS_DEFAULT,
                t_start: float = T_START_DEFAULT,
                n: int = N_DEFAULT) -> pd.Series:
    """
    Convert a time column (in seconds) to sample indices based on sampling rate and window.

    Parameters
    ----------
    series : pd.Series
        Time values in seconds.
    fs : float
        Sampling frequency in Hz.
    t_start : float
        Start time of the window (s).
    n : int
        Total number of samples in the window.

    Returns
    -------
    pd.Series
        Sample indices (float), with NaN for invalid or out-of-window values.
    """
    s = pd.to_numeric(series, errors="coerce")
    idx = np.rint((s - t_start) * fs)
    # Keep NaN for invalid/missing
    idx[~np.isfinite(s)] = np.nan
    # Out-of-window -> NaN
    idx[(idx < 0) | (idx > (n - 1))] = np.nan
    return idx.astype(float)


def apply_time_indexing(
    filtered_sessions: List[Dict[str, pd.DataFrame]],
    task_key: str = "delsac",
    time_cols: Optional[List[str]] = None,
    fs: float = FS_DEFAULT,
    t_start: float = T_START_DEFAULT,
    n: int = N_DEFAULT,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Apply time→index conversion to specified columns for each session, then stack all sessions.

    Parameters
    ----------
    filtered_sessions : list of dict
        Output of `select_correct_trials` (from preprocess.py).
    task_key : str
        Name of the DataFrame key within each session dict.
    time_cols : list of str or None
        Columns to convert. If None, uses TIME_COLS_DEFAULT.
    fs, t_start, n : float
        Sampling parameters for conversion.
    dropna : bool
        Whether to drop rows with NaNs after conversion.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of all sessions with *_idx columns added.
    """
    if time_cols is None:
        time_cols = TIME_COLS_DEFAULT

    processed_sessions = []

    for sess in filtered_sessions:
        if task_key not in sess:
            continue

        df = sess[task_key]
        df = df.copy()

        for col in time_cols:
            if col in df.columns:
                df[f"{col}_idx"] = time_to_idx(df[col], fs=fs, t_start=t_start, n=n)

        if dropna:
            df = df.dropna().copy()

        processed_sessions.append({task_key: df})

    # Concatenate into one DataFrame
    df_all = pd.concat(
        [sess[task_key] for sess in processed_sessions if task_key in sess],
        ignore_index=False
    )
    return df_all


# Session cleanup

def unwrap_scalar(x):
    """Recursively unwrap 0-d arrays or 1-length containers."""
    while isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(x, dtype=object)
        if arr.shape == ():           # 0-d np.array
            x = arr.item()
        elif len(arr) == 1:           # 1-length list/array/Series
            x = arr[0]
        else:
            break                     # keep as-is if multi-length
    return x


def normalize_session(x):
    """
    Normalize a session identifier by stripping array-like brackets and extensions.

    Examples
    --------
    '[session001.npy]' → 'session001'
    'session002.abc' → 'session002'
    """
    x = unwrap_scalar(x)
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    # Remove [ ] brackets and quotes
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1].strip().strip("'").strip('"')

    # Remove file extension if any
    if '.' in s:
        s = s.split('.')[0]
    return s


def add_session_id_column(df: pd.DataFrame, source_col: str = "session") -> pd.DataFrame:
    """
    Add a clean 'session_id' column to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'session' column (or equivalent).
    source_col : str
        Column to use as the session source.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'session_id' column.
    """
    df = df.copy()
    if source_col in df.columns:
        df["session_id"] = df[source_col].apply(normalize_session)
    else:
        df["session_id"] = np.nan
    return df


#  Parsing & normalization utilities

def parse_target_pos(val) -> Tuple[float, float]:
    """
    Parse a 'targetPos' value into (eccentricity_deg, angle_deg).
    Handles tuples/lists/arrays and messy string forms like:
    '(np.uint16(6), np.int16(180))', 'array([6, 180])', etc.
    Returns (np.nan, np.nan) if parsing fails.
    """
    # direct tuple/list/array
    if isinstance(val, (tuple, list, np.ndarray)):
        if len(val) >= 2:
            try:
                return float(val[0]), float(val[1])
            except Exception:
                pass

    # strings with wrappers
    if isinstance(val, str):
        s = val.strip()

        # 1) strip numpy scalar wrappers: np.uint16(6) -> 6
        s = re.sub(r"np\.\w+\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", r"\1", s)

        # 2) strip array wrappers: array([...]) / np.array([...]) -> [...]
        s = re.sub(r"(?:np\.)?array\(\s*(.*?)\s*\)", r"\1", s)

        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (tuple, list)) and len(obj) >= 2:
                return float(obj[0]), float(obj[1])
        except Exception:
            # fallback: strict number extraction (avoid letters before numbers)
            nums = re.findall(r"(?<![A-Za-z])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])

    return (np.nan, np.nan)


def angle_norm_deg(a) -> np.ndarray:
    """
    Normalize degrees to (-180, 180], vectorized.
    Works with NumPy arrays, lists, or pandas Series.
    Preserves NaN.
    """
    # Convert to numeric numpy array safely
    a = np.asarray(pd.to_numeric(a, errors="coerce"), dtype=float)

    out = (a + 180.0) % 360.0 - 180.0

    # Map -180 → 180 while keeping NaNs intact
    mask = np.isfinite(out) & (out == -180.0)
    out[mask] = 180.0

    return out



# High-level helpers to apply on a DataFrame 

def parse_target_columns(
    df: pd.DataFrame,
    source_col: str = "targetPos",
    ecc_col: str = "targEcc",
    ang_col: str = "targAng"
) -> pd.DataFrame:
    """
    Parse `source_col` into clean eccentricity/angle columns (degrees), with angle normalized.
    Returns a copy of df with new/overwritten `ecc_col` and `ang_col`.
    """
    df = df.copy()
    polars = df[source_col].apply(parse_target_pos) if source_col in df.columns else []
    if len(polars):
        # assign parsed polar values
        tmp = pd.DataFrame(polars.tolist(), index=df.index, columns=[ecc_col, ang_col])
        df[[ecc_col, ang_col]] = tmp[[ecc_col, ang_col]]
        # normalize angles
        df[ang_col] = angle_norm_deg(df[ang_col].values)
    else:
        # ensure columns exist even if source missing
        df[ecc_col] = np.nan
        df[ang_col] = np.nan
    return df


def polar_to_cartesian(
    df: pd.DataFrame,
    ecc_col: str = "targEcc",
    ang_col: str = "targAng",
    x_col: str = "targX",
    y_col: str = "targY"
) -> pd.DataFrame:
    """
    Convert polar (deg, deg) to Cartesian (dva) assuming:
      - radius in degrees of visual angle
      - 0° = rightward, +90° = upward (standard math convention)
    Adds/overwrites `x_col`, `y_col`. Returns a copy.
    """
    df = df.copy()
    ecc = pd.to_numeric(df.get(ecc_col, np.nan), errors="coerce")
    ang = pd.to_numeric(df.get(ang_col, np.nan), errors="coerce")
    rad = np.deg2rad(ang)
    df[x_col] = ecc * np.cos(rad)
    df[y_col] = ecc * np.sin(rad)
    return df

# ----- session/subject utilities ----- #


def _unwrap_scalar(x: Any) -> Any:
    """Recursively unwrap 0-d arrays and 1-length containers to a scalar."""
    while isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(x, dtype=object)
        if arr.shape == ():      # 0-d array
            x = arr.item()
        elif len(arr) == 1:      # 1-length container
            x = arr[0]
        else:
            break
    return x

def _clean_subject(x: Any) -> str:
    """
    Normalize subject value to a lowercase string like 'paula'/'rex'.
    Handles 1-elem lists, ndarrays, Series, and stringified lists "['paula']".
    """
    x = _unwrap_scalar(x)

    # If it's a string that looks like a list, parse it
    if isinstance(x, str) and x.strip().startswith('[') and x.strip().endswith(']'):
        try:
            parsed = ast.literal_eval(x)
            x = _unwrap_scalar(parsed)
        except Exception:
            pass

    return str(x).strip().strip('"').strip("'").lower()

def _clean_session(x: Any) -> str:
    """
    Normalize session to a plain string ID.
    Mirrors original behavior: .item() then scalar unwrap, then str.
    """
    # try .item() like your original notebook
    try:
        x = x.item()
    except Exception:
        pass
    x = _unwrap_scalar(x)
    return str(x)

def extract_session_subject_map(data: Dict, task_key: str = "delsac") -> Tuple[pd.DataFrame, List[str], List[str]]:
    rows: List[Dict[str, str]] = []
    for i, sess in enumerate(data.get("session", [])):
        try:
            if not isinstance(sess, dict) or task_key not in sess:
                continue
            block = sess[task_key]
            subject_raw = block.get("subject")
            session_raw = block.get("session")

            # Your original pattern was: subject.item()[0] and session.item()
            # We emulate that robustly:
            subject = _clean_subject(subject_raw)
            session_id = _clean_session(session_raw)

            rows.append({"session_id": session_id, "subject": subject})
        except Exception as e:
            print(f"[warn] Could not extract info for session {i}: {e}")
            continue

    df = pd.DataFrame(rows)
    paula_sessions = df.loc[df["subject"] == "paula", "session_id"].astype(str).tolist()
    rex_sessions   = df.loc[df["subject"] == "rex",   "session_id"].astype(str).tolist()
    return df, paula_sessions, rex_sessions


# ----- .mat file utilities ----- #


def find_session_mat_files(
    base_dir: str | Path,
    sessions: Iterable[str],
    suffix: str = ".mat"
) -> Tuple[List[Path], List[str]]:
    """
    Given a directory and a set of session IDs, return the list of matching .mat files
    and the list of sessions that were missing.

    Parameters
    ----------
    base_dir : str | Path
        Directory containing the .mat files.
    sessions : iterable of str
        Session IDs to look for (should match filenames without extension).
    suffix : str
        File extension to search for (default '.mat').

    Returns
    -------
    files : list of Path
        Paths to matched files, one per existing session.
    missing : list of str
        Session IDs that were not found in the directory.
    """
    base = Path(base_dir)
    want = {str(s) for s in sessions}
    # map stem -> path for all .mat files present
    available = {p.stem: p for p in base.glob(f"*{suffix}") if p.is_file()}
    files = [available[s] for s in want if s in available]
    missing = [s for s in want if s not in available]
    # keep deterministic order (by session id)
    files.sort(key=lambda p: p.stem)
    return files, missing


def load_mat_dict(
    files: Iterable[str | Path],
    squeeze_me: bool = True,
    struct_as_record: Optional[bool] = None,
    simplify_cells: Optional[bool] = True
) -> Dict[str, dict]:
    """
    Load a collection of .mat files into a dict keyed by session_id (filename stem).

    Parameters
    ----------
    files : iterable of str | Path
        File paths to .mat files.
    squeeze_me : bool
        Passed to scipy.io.loadmat (default True).
    struct_as_record : bool | None
        Deprecated in newer SciPy; leave None to use default.
    simplify_cells : bool | None
        If available (SciPy >=1.7), makes cell arrays become lists.

    Returns
    -------
    mat_data : dict
        {session_id: loaded_mat_dict}
    """
    out: Dict[str, dict] = {}
    for f in files:
        f = Path(f)
        kwargs = {"squeeze_me": squeeze_me}
        if struct_as_record is not None:
            kwargs["struct_as_record"] = struct_as_record  # older SciPy compatibility
        # Some SciPy versions don’t support simplify_cells
        try:
            if simplify_cells is not None:
                kwargs["simplify_cells"] = simplify_cells  # type: ignore[arg-type]
            md = scipy.io.loadmat(str(f), **kwargs)
        except TypeError:
            # retry without simplify_cells if not supported
            kwargs.pop("simplify_cells", None)
            md = scipy.io.loadmat(str(f), **kwargs)
        out[f.stem] = md
    return out


def count_total_trials(
    mat_data: Dict[str, dict],
    array_key: str = "ain",
    trial_axis: int = 2
) -> int:
    """
    Sum trial counts across sessions by inspecting the shape of `array_key`.

    Parameters
    ----------
    mat_data : dict
        {session_id: loaded_mat_dict}
    array_key : str
        Key inside each .mat dict with shape (..., n_trials) along `trial_axis`.
    trial_axis : int
        Axis index holding trials.

    Returns
    -------
    total : int
        Sum of trials over all sessions where `array_key` exists.
    """
    total = 0
    for sid, md in mat_data.items():
        if array_key in md and isinstance(md[array_key], np.ndarray):
            arr = md[array_key]
            if arr.ndim > trial_axis:
                total += arr.shape[trial_axis]
    return total




# ---------- tiny helpers (robust unwrapping / walking nested object arrays) ----------

def peel(x: Any) -> Any:
    """Recursively unwrap 0-d/1-elem object ndarrays to a scalar."""
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.reshape(-1)[0]
    return x

def walk(obj: Any):
    """Depth-first iteration over leaves in nested lists/tuples/object ndarrays."""
    obj = peel(obj)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for y in obj.ravel():
            yield from walk(y)
    elif isinstance(obj, (list, tuple)):
        for y in obj:
            yield from walk(y)
    else:
        yield obj

# ---------- schema → trial numbers ----------

def get_trial_nums_from_schema(ain_schema_index: Any, expected_len: int) -> np.ndarray:
    """
    Extract trial numbers (shape T,) from MATLAB-loaded ainSchema.index (messy nested structs).
    Tries the common layout [time, channels, trials], else searches any int array of length T.
    """
    idx = peel(ain_schema_index)
    # Try canonical [time, channels, trials]
    try:
        cand = peel(idx)[2]
        arr = np.asarray(peel(cand)).astype(int).ravel()
        if arr.size == expected_len:
            return arr
    except Exception:
        pass
    # Fallback: walk and find an int array of the right length
    for leaf in walk(idx):
        if isinstance(leaf, np.ndarray) and leaf.dtype.kind in "iu":
            arr = np.asarray(leaf).astype(int).ravel()
            if arr.size == expected_len:
                return arr
    raise ValueError("Couldn't find trial numbers in ainSchema.index")

# ---------- df_idx trial frame construction ----------

def _extract_session_id_from_index_value(v: Any) -> str:
    """Best-effort conversion of a possibly-arraylike index value to 'session_id' string (no extension)."""
    try:
        arr = np.asarray(v)
        s = arr.flat[0] if arr.shape else v
    except Exception:
        s = v
    return str(s).split('.')[0]

def make_trial_frame(
    df_idx: pd.DataFrame,
    session_col: str = "session_id",
    trial_col: str = "trial"
) -> pd.DataFrame:
    """
    Produce a tidy frame with columns ['session_id','trial'] as Int64,
    preserving the original row order of df_idx.
    If session_col not present, tries to infer it from the index.
    """
    if session_col in df_idx.columns:
        df = df_idx[[session_col, trial_col]].copy()
        df[session_col] = df[session_col].astype(str)
    else:
        # Reset index and take the first extra column as the session identifier (like your notebook)
        df_r = df_idx.reset_index(drop=False).copy()
        extra_cols = [c for c in df_r.columns if c not in df_idx.columns]
        idx_col = extra_cols[0] if extra_cols else df_r.columns[0]
        df = pd.DataFrame({
            session_col: df_r[idx_col].apply(_extract_session_id_from_index_value).astype(str),
            trial_col: df_r.get(trial_col)
        })
    df[trial_col] = pd.to_numeric(df[trial_col], errors="coerce").astype("Int64")
    df = df[df[trial_col].notna()].copy()
    return df

# ---------- core: filter MAT data by df trials (order-preserving) ----------

def filter_mat_data_by_trials(
    df_idx: pd.DataFrame,
    mat_data_dict: Dict[str, dict],
    ain_key: str = "ain",
    schema_key: str = "ainSchema",
    keep_trials_key: str = "kept_trial_ids"
) -> Tuple[Dict[str, dict], pd.DataFrame]:
    """
    For each session present in both df_idx and mat_data_dict:
      - read trial numbers from ainSchema
      - map df trials to positions along MAT array trial axis
      - slice ain accordingly (preserving df order)
      - carry through other fields unchanged
      - store the kept original trial IDs under `keep_trials_key`

    Returns
    -------
    filtered_dict : {session_id: {ain: sliced, keep_trials_key: np.ndarray[int], ...}}
    summary_df    : DataFrame with columns ['session_id','trials_total','trials_kept']
    """
    tf = make_trial_frame(df_idx)  # ['session_id','trial']
    # ensure string session ids for matching
    tf["session_id"] = tf["session_id"].astype(str)

    common_sessions = sorted(set(mat_data_dict.keys()) & set(tf["session_id"].unique().astype(str)))
    filtered: Dict[str, dict] = {}
    summary_rows: List[Tuple[str, int, int]] = []

    for sess_id in common_sessions:
        sess = mat_data_dict[sess_id]
        if ain_key not in sess:
            summary_rows.append((sess_id, 0, 0))
            continue

        ain = sess[ain_key]  # expected shape (S, C, T)
        if not isinstance(ain, np.ndarray) or ain.ndim < 3:
            raise ValueError(f"{sess_id}: '{ain_key}' must be ndarray with >=3 dims, got {type(ain)} shape={getattr(ain,'shape',None)}")

        S, C, T = ain.shape[-3], ain.shape[-2], ain.shape[-1]

        schema = sess.get(schema_key, None)
        if schema is None:
            raise KeyError(f"{sess_id}: missing '{schema_key}'")

        # ainSchema can be struct array; typical access rec = schema[0,0]
        rec = schema[0, 0] if isinstance(schema, np.ndarray) else schema
        trial_nums = get_trial_nums_from_schema(rec["index"], T)   # shape (T,)
        lookup = {int(t): i for i, t in enumerate(trial_nums)}     # trial_id -> axis pos

        # Trials for this session in df order
        trials_df_ordered = tf.loc[tf["session_id"] == sess_id, "trial"].astype(int).tolist()
        pos = [lookup[t] for t in trials_df_ordered if t in lookup]

        if len(pos) == 0:
            summary_rows.append((sess_id, T, 0))
            continue

        # Slice last axis (trial axis), preserving order
        slicer = [slice(None)] * ain.ndim
        slicer[-1] = pos
        ain_f = ain[tuple(slicer)]
        kept_trial_ids = np.asarray([trial_nums[i] for i in pos], dtype=int)

        out = {ain_key: ain_f, keep_trials_key: kept_trial_ids}
        # carry other keys unchanged
        for k, v in sess.items():
            if k not in (ain_key, keep_trials_key):
                out[k] = v
        filtered[sess_id] = out

        summary_rows.append((sess_id, T, ain_f.shape[-1]))

    summary_df = pd.DataFrame(summary_rows, columns=["session_id", "trials_total", "trials_kept"])
    return filtered, summary_df


def _angle_deg(x, y):
    """
    Return angle in degrees wrapped to (-180, 180].
    Works with scalars or numpy arrays.
    """
    ang = np.degrees(np.arctan2(y, x))
    return (ang + 180.0) % 360.0 - 180.0


def add_response_metrics(df_trials: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """
    Per session:
      - Fit a 2x2 map A on correct trials in baseline-relative space (no intercept).
      - Compute response metrics in raw baseline-relative coords (respEcc/respAng).
      - Transform responses to dva via Ainv, and compute respX/Y_dva + respEcc/Ang_dva.

    Parameters
    ----------
    df_trials : DataFrame
        Must contain columns:
        ['session_id','correct','targX','targY','endX_raw','endY_raw','fixBaseX_raw','fixBaseY_raw']

    Returns
    -------
    df_out : DataFrame
        Copy of input with added columns:
        ['respEcc_raw','respAng_raw','respX_dva','respY_dva','respEcc_dva','respAng_dva']
    per_session_fit : dict
        Mapping session_id -> {'A': 2x2 ndarray or None, 'Ainv': 2x2 ndarray or None}
    """
    df = df_trials.copy().reset_index(drop=True)
    if "session_id" in df.columns:
        df["session_id"] = df["session_id"].astype(str)

    out_cols = ["respEcc_raw","respAng_raw","respX_dva","respY_dva","respEcc_dva","respAng_dva"]
    for col in out_cols:
        df[col] = np.nan

    per_session_fit: Dict[str, Dict[str, np.ndarray]] = {}

    needed = ["correct","targX","targY","endX_raw","endY_raw","fixBaseX_raw","fixBaseY_raw"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"add_response_metrics_big: missing required columns: {missing}")

    # Group by session and process
    for sid, rows in df.groupby("session_id", sort=False):
        idx = rows.index

        good = (rows["correct"] == 1) & rows[needed].notna().all(axis=1)

        R_all = rows[["endX_raw","endY_raw"]].to_numpy(float)
        B_all = rows[["fixBaseX_raw","fixBaseY_raw"]].to_numpy(float)
        R_all_rel = R_all - B_all

        # Raw (baseline-relative) response metrics
        respEcc_raw = np.hypot(R_all_rel[:, 0], R_all_rel[:, 1])
        respAng_raw = _angle_deg(R_all_rel[:, 0], R_all_rel[:, 1])

        # Defaults for dva
        respX_dva = np.full(len(rows), np.nan, float)
        respY_dva = np.full(len(rows), np.nan, float)

        A = None
        Ainv = None

        if np.any(good):
            T = rows.loc[good, ["targX","targY"]].to_numpy(float)               # (N x 2)
            R = rows.loc[good, ["endX_raw","endY_raw"]].to_numpy(float)         # (N x 2)
            B = rows.loc[good, ["fixBaseX_raw","fixBaseY_raw"]].to_numpy(float) # (N x 2)
            R_rel = R - B

            A, *_ = np.linalg.lstsq(T, R_rel, rcond=None)  # (2x2)
            Ainv = np.linalg.pinv(A)

            # Transform all responses into dva coords
            respXY_dva = R_all_rel @ Ainv
            respX_dva[:] = respXY_dva[:, 0]
            respY_dva[:] = respXY_dva[:, 1]

        # Write back to main df
        df.loc[idx, "respEcc_raw"] = respEcc_raw
        df.loc[idx, "respAng_raw"] = respAng_raw
        df.loc[idx, "respX_dva"]   = respX_dva
        df.loc[idx, "respY_dva"]   = respY_dva
        df.loc[idx, "respEcc_dva"] = np.hypot(respX_dva, respY_dva)
        df.loc[idx, "respAng_dva"] = _angle_deg(respX_dva, respY_dva)

        per_session_fit[str(sid)] = {"A": A, "Ainv": Ainv}

    return df, per_session_fit


def circ_diff(a: float, b: float) -> float:
    """Circular difference (a - b) wrapped to [-180, 180]."""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return (a - b + 180) % 360 - 180


def circ_err(curr: float, resp: float) -> float:
    """Circular error (curr - resp) wrapped to [-180, 180]."""
    if pd.isna(curr) or pd.isna(resp):
        return np.nan
    return (curr - resp + 180) % 360 - 180


def add_memory_delay_features(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Add memory-delay and circular metrics:
      - memoryDelay  = responseTime - targetOff
      - prev         = previous trial's target angle
      - diff         = |circular(prev - curr)|  (change in target direction)
      - err          = circular(curr - resp)   (angular response error)

    Parameters
    ----------
    merged : pd.DataFrame
        Must include columns:
        ['responseTime','targetOff','targAng','respAng_dva']

    Returns
    -------
    df : pd.DataFrame
        Copy of merged with added columns:
        ['memoryDelay','prev','diff','err']
    """
    df = merged.copy()

    required = ["responseTime", "targetOff", "targAng", "respAng_dva"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for add_memory_delay_features: {missing}")

    # Memory delay (in same time units as input)
    df["memoryDelay"] = df["responseTime"] - df["targetOff"]

    # Current, previous, and response angles
    df["curr"] = df["targAng"]
    df["resp"] = df["respAng_dva"]
    df["prev"] = df["curr"].shift(1)

    # Circular difference between prev and curr (absolute)
    df["diff"] = [abs(circ_diff(p, c)) for p, c in zip(df["prev"], df["curr"])]

    # Circular error (curr - resp)
    df["err"] = [circ_err(c, r) for c, r in zip(df["curr"], df["resp"])]

    return df


def compute_features_for_session(
    ain: np.ndarray,
    idx_tbl: pd.DataFrame,
    pre_go_win: int = 25,
    halfwin: int = 10
) -> pd.DataFrame:
    """
    ain: (S, C>=2, T) with ch0=X, ch1=Y
    idx_tbl: rows aligned to trial axis order (0..T-1)
    returns DataFrame with fixBaseX_raw, fixBaseY_raw, endX_raw, endY_raw
            plus targX, targY, correct, response indices
    """
    S, C, T = ain.shape
    assert C >= 2, "AIN must have at least two channels (X=0, Y=1)."

    endX = np.full(T, np.nan, float)
    endY = np.full(T, np.nan, float)
    baseX = np.full(T, np.nan, float)
    baseY = np.full(T, np.nan, float)

    for i in range(T):
        # Endpoint around responseDone
        t_done = idx_tbl.at[i, "responseDone_idx"]
        if pd.notna(t_done):
            c = _clip_idx_small(t_done, S)
            s, e = _endpoint_window(c, halfwin, S)
            x = ain[s:e, 0, i]
            y = ain[s:e, 1, i]
            if x.size:
                endX[i] = float(np.nanmean(x))
                endY[i] = float(np.nanmean(y))

        # Baseline before fixptOff
        t_go = idx_tbl.at[i, "fixptOff_idx"]
        if pd.notna(t_go):
            e2 = _clip_idx_small(t_go, S)
            s2, e2b = _baseline_window(e2, pre_go_win, S)
            bx = ain[s2:e2b, 0, i]
            by = ain[s2:e2b, 1, i]
            if bx.size:
                baseX[i] = float(np.nanmean(bx))
                baseY[i] = float(np.nanmean(by))

    # NOTE: keep 'correct' as pandas nullable Int64; use .to_numpy(float) for others
    out = pd.DataFrame({
        "fixBaseX_raw": baseX,
        "fixBaseY_raw": baseY,
        "endX_raw": endX,
        "endY_raw": endY,
        "targX": idx_tbl["targX"].to_numpy(float),
        "targY": idx_tbl["targY"].to_numpy(float),
        "correct": idx_tbl["correct"].astype("Int64"),  # <- no .to_numpy("Int64")
        "responseTime_idx": idx_tbl["responseTime_idx"].to_numpy(float),
        "responseDone_idx": idx_tbl["responseDone_idx"].to_numpy(float),
        "fixptOff_idx": idx_tbl["fixptOff_idx"].to_numpy(float),
    })
    return out




