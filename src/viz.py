from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


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
    bins: int = 60,
    bin_range: tuple[float, float] = (-20, 20),
    color: str = "steelblue",
    alpha: float = 0.8,
    figsize: tuple[int, int] = (7, 5),
    title: str = "Histogram of saccade angular error",
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a histogram of saccade angular error (in degrees).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the error column.
    err_col : str, optional
        Column name for angular error (default 'err').
    bins : int, optional
        Number of histogram bins (default 60).
    bin_range : tuple, optional
        Range of the histogram in degrees (default (-20, 20)).
    color : str, optional
        Bar color (default 'steelblue').
    alpha : float, optional
        Transparency for bars (default 0.8).
    figsize : tuple[int, int], optional
        Figure size in inches (default (7, 5)).
    title : str, optional
        Plot title.
    show : bool, optional
        Whether to display the plot immediately (default True).
    save_path : str | None, optional
        If provided, saves the figure to the given path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    if err_col not in df.columns:
        raise ValueError(f"Column '{err_col}' not found in DataFrame.")

    err = df[err_col].dropna()

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(err, bins=bins, range=bin_range, color=color, alpha=alpha, edgecolor="k")

    ax.set_xlabel("Error (curr − resp) [deg]")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.axvline(0, color="red", linestyle="--", label="perfect alignment")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # return fig



def plot_error_variability_vs_delay(
    df: pd.DataFrame,
    delay_col: str = "delay",
    err_col: str = "err",
    bins: int = 200,
    figsize: tuple[int, int] = (7, 5),
    color: str = "steelblue",
    alpha: float = 0.8,
    show: bool = True,
    save_path: str | None = None,
    title: str = "Error variability vs. memory delay",
) -> plt.Figure:
    """
    Plot the relationship between memory delay duration and variability in angular error.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `delay` and `err` columns (from memory-delay features).
    delay_col : str, optional
        Column name for memory delay (default "delay").
    err_col : str, optional
        Column name for angular error (default "err").
    bins : int, optional
        Number of quantile bins to divide the delay range (default 200).
    figsize : tuple[int, int], optional
        Size of the figure (default (7,5)).
    color : str, optional
        Point color for scatter (default "steelblue").
    alpha : float, optional
        Transparency of scatter points (default 0.8).
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

    valid = df[[delay_col, err_col]].dropna().copy()

    # Bin delay into quantiles
    valid["delay_bin"] = pd.qcut(valid[delay_col], q=bins, duplicates="drop")

    # Compute per-bin means and stds
    stats = valid.groupby("delay_bin", observed=False).agg(
        delay_mean=(delay_col, "mean"),
        err_std=(err_col, "std")
    ).reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(stats["delay_mean"], stats["err_std"],
               color=color, s=60, alpha=alpha, label="binned std")

    # Fit and plot linear trend if enough points
    if len(stats) > 2:
        coeffs = np.polyfit(stats["delay_mean"], stats["err_std"], 1)
        xs = np.linspace(stats["delay_mean"].min(), stats["delay_mean"].max(), 100)
        ys = np.polyval(coeffs, xs)
        ax.plot(xs, ys, color="red", label=f"fit: slope={coeffs[0]:.2f} std/s")

    ax.set_xlabel("Memory delay (s)")
    ax.set_ylabel("Std of error (deg)")
    ax.set_title(title)
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
    err_col: str = "err",
    bin_size: int = 30,
    figsize: tuple[int, int] = (8, 5),
    color: str = "steelblue",
    show: bool = True,
    save_path: str | None = None,
    verbose: bool = False,
    title: str = "Mean error vs. |Prev − Curr| angular difference",
) -> plt.Figure:
    """
    Plot mean ± SEM of angular error as a function of the absolute difference
    between previous and current target directions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns `diff` and `err` (from memory-delay features).
    diff_col : str, optional
        Column name for angular difference (default 'diff').
    err_col : str, optional
        Column name for angular error (default 'err').
    bin_size : int, optional
        Bin width in degrees (default 30°).
    figsize : tuple[int, int], optional
        Figure size (default (8,5)).
    color : str, optional
        Line/marker color (default "steelblue").
    show : bool, optional
        Whether to display the plot immediately (default True).
    save_path : str | None, optional
        If provided, saves the figure to the given path.
    verbose : bool, optional
        If True, print per-bin debug information (default False).
    title : str, optional
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib figure.
    """
    if diff_col not in df.columns or err_col not in df.columns:
        raise ValueError(f"Missing required columns: {diff_col}, {err_col}")

    valid = df[[diff_col, err_col]].dropna().copy()

    # Define bin edges
    bin_edges = np.arange(-180, 181, bin_size)
    valid["diff_bin"] = pd.cut(valid[diff_col], bins=bin_edges)

    if verbose:
        print("Diff bins:")
        print(valid["diff_bin"].value_counts().sort_index())

    # Compute per-bin stats
    stats = valid.groupby("diff_bin", observed=False)[err_col].agg(["mean", "count", "std"])
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])

    # Bin centers for plotting
    bin_centers = [interval.mid for interval in stats.index]

    if verbose:
        for center, interval in zip(bin_centers, stats.index):
            bin_err_values = valid.loc[valid["diff_bin"] == interval, err_col].tolist()
            bin_diff_values = valid.loc[valid["diff_bin"] == interval, diff_col].tolist()
            print(f"Bin Center: {center}, Bin Range: {interval}, "
                  f"Diff Values: {bin_diff_values}, Err Values: {bin_err_values}")

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        bin_centers,
        stats["mean"],
        yerr=stats["sem"],
        fmt="o-",
        capsize=4,
        color=color,
        label="mean ± SEM"
    )

    ax.axhline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("abs(Prev − Curr) (deg, binned)")
    ax.set_ylabel("Error (deg)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # return fig


