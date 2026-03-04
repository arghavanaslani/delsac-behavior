from __future__ import annotations
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import os
import math


def plot_sessions(
    merged: pd.DataFrame,
    output_dir: str = "session_plots",
    figsize: tuple[int, int] = (7, 7),
    save: bool = True,
    show: bool = False,
    verbose: bool = True,
) -> None:
    """
    Generate per-session scatter plots showing fixation baselines, endpoints,
    and predicted targets. Saves each session’s figure as PNG.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged dataframe containing at least:
        ['fixBaseX_raw','fixBaseY_raw','endX_raw','endY_raw',
         'targX_raw_pred','targY_raw_pred','session_id']
    output_dir : str, optional
        Directory where plots will be saved (default 'session_plots').
    figsize : tuple[int, int], optional
        Figure size in inches (default (7,7)).
    save : bool, optional
        Whether to save each session’s plot as a PNG file (default True).
    show : bool, optional
        Whether to display each plot interactively (default False).
    verbose : bool, optional
        Print progress info (default True).
    """
    req = [
        "fixBaseX_raw","fixBaseY_raw",
        "endX_raw","endY_raw",
        "targX_raw_pred","targY_raw_pred",
    ]

    # Verify required columns
    missing = [c for c in req if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure directory exists
    if save:
        os.makedirs(output_dir, exist_ok=True)

    # Unique session IDs
    sessions = sorted(merged["session_id"].astype(str).unique())

    for sid in sessions:
        sess = merged.loc[merged["session_id"].astype(str) == sid].copy()
        mask = sess[req].notna().all(axis=1)
        n_points = int(mask.sum())

        if verbose:
            print(f"Session {sid}: {n_points} valid trials.")

        if n_points == 0:
            continue  # skip empty sessions

        fig, ax = plt.subplots(figsize=figsize)

        # Fixation baselines (green)
        ax.scatter(
            sess.loc[mask, "fixBaseX_raw"],
            sess.loc[mask, "fixBaseY_raw"],
            c="green", marker="o", s=40, alpha=0.5, label="Fixation baseline"
        )

        # Endpoints (blue)
        ax.scatter(
            sess.loc[mask, "endX_raw"],
            sess.loc[mask, "endY_raw"],
            c="blue", s=40, alpha=0.6, label="Saccade endpoints"
        )

        # Predicted targets (red)
        ax.scatter(
            sess.loc[mask, "targX_raw_pred"],
            sess.loc[mask, "targY_raw_pred"],
            c="red", marker="x", s=80, label="Predicted targets"
        )

        ax.set_title(f"Session {sid} (n={n_points})")
        ax.set_xlabel("X (raw tracker units)")
        ax.set_ylabel("Y (raw tracker units)")
        ax.axis("equal")
        ax.legend()
        plt.tight_layout()

        if save:
            path = os.path.join(output_dir, f"{sid}.png")
            fig.savefig(path, dpi=150)
            if verbose:
                print(f"  → Saved to {path}")

        if show:
            plt.show()
        else:
            plt.close(fig)



def plot_eye_endpoints(
    merged: pd.DataFrame,
    figsize: tuple[int, int] = (7, 7),
    show: bool = True,
    save_path: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Plot fixation baselines, saccade endpoints, and predicted targets.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged dataframe containing columns:
        ['endX_raw','endY_raw','targX_raw_pred','targY_raw_pred',
         'responseTime_idx','responseDone_idx','fixBaseX_raw','fixBaseY_raw']
    figsize : tuple[int, int], optional
        Figure size in inches (default: (7,7)).
    show : bool, optional
        Whether to display the plot immediately (default: True).
    save_path : str | None, optional
        If given, save the figure to this path instead of (or in addition to) showing it.
    title : str | None, optional
        Optional figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object (for further customization or saving).
    """
    req = [
        "endX_raw", "endY_raw",
        "targX_raw_pred", "targY_raw_pred",
        "responseTime_idx", "responseDone_idx",
        "fixBaseX_raw", "fixBaseY_raw",
    ]

    # Check columns
    missing = [c for c in req if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    mask = merged[req].notna().all(axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    # Fixation baselines (green circles)
    ax.scatter(
        merged.loc[mask, "fixBaseX_raw"],
        merged.loc[mask, "fixBaseY_raw"],
        c="green", marker="o", s=40, alpha=0.5, label="Fixation point (raw)"
    )

    # Endpoints (blue dots)
    ax.scatter(
        merged.loc[mask, "endX_raw"],
        merged.loc[mask, "endY_raw"],
        c="blue", s=40, alpha=0.6, label="Saccade endpoints"
    )

    # Predicted targets (red crosses)
    ax.scatter(
        merged.loc[mask, "targX_raw_pred"],
        merged.loc[mask, "targY_raw_pred"],
        c="red", marker="x", s=80, label="Predicted targets"
    )

    ax.set_xlabel("X (tracker units)")
    ax.set_ylabel("Y (tracker units)")
    ax.axis("equal")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # return fig





def plot_error_histogram(
    df: pd.DataFrame,
    err_col: str = "err",
    bins: int = 1000,
    bin_range: tuple[float, float] = (-180, 180),
    color: str = "k",
    alpha: float = 0.8,
    figsize: tuple[int, int] = (7, 4),
    title: str = "Angular Error Distribution",
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot histogram of error values with probability density on the y-axis.
    """

    if err_col not in df.columns:
        raise ValueError(f"Column '{err_col}' not found in DataFrame.")

    err = df[err_col].dropna()

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram normalized to probability density
    ax.hist(
        err,
        bins=bins,
        range=bin_range,
        color=color,
        alpha=alpha,
        density=True,
    )

    ax.set_xlabel("Error (deg)", fontsize=20)
    ax.set_ylabel("Probability", fontsize=20)
    ax.set_title(title, fontsize=22)

    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([0, 0.05, 0.1])

    # Vertical line at zero error
    ax.axvline(0, color="red", linestyle="--", label="perfect alignment")
    ax.legend(fontsize=15)

    ax.set_xlim(bin_range)
    
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig



def plot_error_variability_vs_delay(
    df: pd.DataFrame,
    delay_col: str = "delay",
    err_col: str = "abs_err",
    bins: int = 2000,  # deprecated, kept for backward compatibility
    figsize: tuple[int, int] = (7, 5),
    color: str = "steelblue",
    alpha: float = 0.1,
    show: bool = True,
    save_path: str | None = None,
    title: str = "Error variability vs. memory delay",
) -> plt.Figure:
    """
    Plot the relationship between memory delay duration and error magnitude,
    and test whether error tends to increase or decrease with delay.

    This version does NOT bin the data. It uses single-trial data and a
    Spearman rank correlation to assess monotonic trend.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `delay_col` and `err_col` columns.
        `err_col` should reflect error magnitude (e.g. absolute or folded error).
    delay_col : str, optional
        Column name for memory delay (default "delay").
    err_col : str, optional
        Column name for error magnitude (default "folded_err").
    bins : int, optional (deprecated)
        Ignored. Kept only for backward compatibility.
    figsize : tuple[int, int], optional
        Size of the figure (default (7, 5)).
    color : str, optional
        Color for scatter points (default "steelblue").
    alpha : float, optional
        Transparency of scatter points (default 0.3).
    show : bool, optional
        Whether to display the figure immediately (default True).
    save_path : str | None, optional
        If provided, save the plot to this file path.
    title : str, optional
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    if delay_col not in df.columns or err_col not in df.columns:
        raise ValueError(f"Missing required columns: {delay_col}, {err_col}")

    # Drop NaNs
    valid = df[[delay_col, err_col]].dropna().copy()

    if valid.empty:
        raise ValueError("No valid (non-NaN) rows for delay and error columns.")

    x = valid[delay_col].to_numpy()
    y = valid[err_col].to_numpy()

    # Spearman rank correlation: does error tend to go up or down with delay?
    rho, pval = spearmanr(x, y)

    # Figure and scatter of raw trials
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color=color, s=20, alpha=alpha, label="single trials")

    # Linear fit (degree 1 polynomial) purely for visualization
    if len(valid) > 2:
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ys = np.polyval(coeffs, xs)
        ax.plot(xs, ys, "r-", linewidth=2,
                label=f"linear fit (slope = {coeffs[0]:.3f})")

    # Labels and title, include Spearman result in subtitle/legend
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Error (deg)")
    ax.set_title(title)

    # Add Spearman info as text in the plot
    ax.text(
        0.05, 0.95,
        f"Spearman ρ = {rho:.3f}\n p = {pval:.3g}",
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # return fig



def plot_error_vs_prev_diff(
    df: pd.DataFrame,
    diff_col: str = "diff",
    err_col: str = "folded_err",
    figsize: tuple[int, int] = (8, 5),
    color: str = "steelblue",
    show: bool = True,
    save_path: str | None = None,
    verbose: bool = False,
    title: str = "Mean error vs. |Prev − Curr| angular difference",
) -> plt.Figure:
    """
    Plot mean ± SEM of angular error as a function of the absolute
    difference between previous and current target directions.

    This version uses **no binning** — each unique absolute diff
    value is treated independently.
    """

    if diff_col not in df.columns or err_col not in df.columns:
        raise ValueError(f"Missing required columns: {diff_col}, {err_col}")

    # Drop NaNs
    valid = df[[diff_col, err_col]].dropna().copy()

    # Use absolute difference
    valid["abs_diff"] = valid[diff_col].abs()

    # Group by each unique abs_diff value exactly
    stats = (
        valid.groupby("abs_diff", observed=False)[err_col]
        .agg(["mean", "count", "std"])
        .reset_index()
        .sort_values("abs_diff")
    )

    # Compute SEM
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])

    if verbose:
        print(stats)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        stats["abs_diff"],
        stats["mean"],
        yerr=stats["sem"],
        fmt="o",
        color=color,
        capsize=3,
        markersize=4,
        label="mean ± SEM (no bins)"
    )
    ax.plot(stats["abs_diff"], stats["mean"], color=color, linewidth=1)

    ax.axhline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("|Prev − Curr| (deg)")
    ax.set_ylabel("Folded Error (deg)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-1,1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # return fig


