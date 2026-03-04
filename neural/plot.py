from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from tqdm import trange,tqdm
import pickle
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import sys, os
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut # YOu can use onc ethe CV works
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
import scipy.signal
import itertools
from math import ceil
import ssm
from ssm.util import find_permutation
from sklearn.model_selection import KFold
import time # in case you want to keep track of time, but be careful with the name of the bins times
from collections import defaultdict
import numpy as np
from astropy.stats import circcorrcoef

def _pearson_r_safe(a, b):
    a = np.asarray(a); b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2: return np.nan
    a = a[m]; b = b[m]
    sa = a.std(ddof=1); sb = b.std(ddof=1)
    if sa == 0 or sb == 0: return np.nan
    za = (a - a.mean()) / sa
    zb = (b - b.mean()) / sb
    return (za @ zb) / (za.size - 1)

angle_pairs = {
    'targetAngle':     ('targetX', 'targetY'),
    'respAngle':       ('respX', 'respY'),
    'prevTargetAngle': ('prevTargetX', 'prevTargetY'),
    'prevRespAngle':   ('prevRespX', 'prevRespY'),
}

def aggregate_circcorrcoef_with_shuffles(all_predictions, all_shuffles, data, angle_pairs):
    """
    angle_pairs: dict mapping angle_name -> (varX, varY)
        e.g. 'targetAngle': ('targetX', 'targetY')

    Uses:
      all_predictions[area][session][var]['predictions'] -> (n_trials, n_time)
      all_shuffles[area][session][var]                   -> (S, n_trials, n_time)
      data['trial'][session][varX / varY]                -> (n_trials,)
    """
    per_sess_real = defaultdict(lambda: defaultdict(list))
    per_sess_null = defaultdict(lambda: defaultdict(list))

    for area in all_predictions.keys():
        for session in all_predictions[area].keys():
            preds_dict = all_predictions[area][session]
            trial_df   = data['trial'][session]

            for angle_name, (varX, varY) in angle_pairs.items():
                # make sure both X and Y were decoded for this session/area
                if varX not in preds_dict or varY not in preds_dict:
                    continue

                predsX = np.asarray(preds_dict[varX]['predictions'])  # (n_trials, n_time)
                predsY = np.asarray(preds_dict[varY]['predictions'])  # (n_trials, n_time)

                # predicted angles (radians)
                pred_angle = np.arctan2(predsY, predsX)               # (n_trials, n_time)

                # true X/Y and angles
                trueX = np.asarray(trial_df[varX])
                trueY = np.asarray(trial_df[varY])
                true_angle = np.arctan2(trueY, trueX)                 # (n_trials,)

                n_trials, n_time = pred_angle.shape

                # mask out NaNs (e.g. first trial for "prev" vars, if you set them to NaN)
                valid_mask = ~np.isnan(true_angle)

                # --- real circular correlation curve ---
                circ_curve = np.full(n_time, np.nan)
                for t in range(n_time):
                    p = pred_angle[valid_mask, t]
                    y = true_angle[valid_mask]
                    if np.all(np.isnan(p)) or np.all(np.isnan(y)):
                        circ_curve[t] = np.nan
                    else:
                        circ_curve[t] = circcorrcoef(p, y)

                per_sess_real[area][angle_name].append(circ_curve)

                # --- shuffled circular correlation curve (mean across shuffles) ---
                circ_null = np.full(n_time, np.nan)
                if (area in all_shuffles and
                    session in all_shuffles[area] and
                    varX in all_shuffles[area][session] and
                    varY in all_shuffles[area][session]):

                    shufsX = np.asarray(all_shuffles[area][session][varX])  # (S, n_trials, n_time)
                    shufsY = np.asarray(all_shuffles[area][session][varY])  # (S, n_trials, n_time)
                    S = shufsX.shape[0]

                    for t in range(n_time):
                        circ_s = np.full(S, np.nan)
                        for s in range(S):
                            pX = shufsX[s, :, t]
                            pY = shufsY[s, :, t]
                            p_angle = np.arctan2(pY[valid_mask], pX[valid_mask])
                            y = true_angle[valid_mask]
                            if np.all(np.isnan(p_angle)) or np.all(np.isnan(y)):
                                circ_s[s] = np.nan
                            else:
                                circ_s[s] = circcorrcoef(p_angle, y)
                        circ_null[t] = np.nanmean(circ_s)

                per_sess_null[area][angle_name].append(circ_null)

    # --- average across sessions ---
    all_mean_scores = defaultdict(dict)
    all_sem_scores  = defaultdict(dict)
    all_null_scores = defaultdict(dict)

    for area in per_sess_real.keys():
        for angle_name in per_sess_real[area].keys():
            real_stack = np.stack(per_sess_real[area][angle_name], axis=0)   # (n_sessions, n_time)
            mean_curve = np.nanmean(real_stack, axis=0)
            n_eff      = np.sum(~np.isnan(real_stack), axis=0)
            sem_curve  = np.nanstd(real_stack, axis=0) / np.where(n_eff > 0, np.sqrt(n_eff), np.nan)

            all_mean_scores[area][angle_name] = mean_curve
            all_sem_scores[area][angle_name]  = sem_curve

            null_stack = np.stack(per_sess_null[area][angle_name], axis=0)
            all_null_scores[area][angle_name] = np.nanmean(null_stack, axis=0)

    return all_mean_scores, all_sem_scores, all_null_scores



def plot_time_resolved_pearson(all_mean_scores, all_sem_scores, time, areas,
                               previous_variables, current_variables,
                               all_null_scores=None):

    variable_pairs = list(zip(previous_variables, current_variables))
    row_labels = ['targetAngle', 'respAngle']
    column_labels = ["Previous", "Current"]

    n_rows = len(variable_pairs)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 12), sharex=True)
    axes = np.atleast_2d(axes)

    area_colors = {'PFC':'b','FEF':'r','LIP':'g','Parietal':'y','IT':'c','MT':'m','V4':'orange'}

    col_ymins = [np.inf for _ in range(n_cols)]
    col_ymaxs = [-np.inf for _ in range(n_cols)]

    for i, (prev_var, curr_var) in enumerate(variable_pairs):
        for j, var in enumerate([prev_var, curr_var]):
            ax = axes[i, j]
            drew_any = False

            for area in areas:
                mean_scores = all_mean_scores.get(area, {})
                sem_scores  = all_sem_scores.get(area, {})
                null_scores = all_null_scores.get(area, {}) if all_null_scores is not None else {}

                if var not in mean_scores:
                    continue

                mean_curve = mean_scores[var]
                sem_curve  = sem_scores[var]
                null_curve = null_scores.get(var, None)

                if mean_curve is not None and not np.isnan(mean_curve).all():
                    ax.plot(time, mean_curve, label=area, linewidth=3,
                            color=area_colors.get(area, 'black'), zorder=2)
                    ax.fill_between(time, mean_curve - sem_curve, mean_curve + sem_curve,
                                    alpha=0.3, color=area_colors.get(area, 'black'), zorder=1)
                    col_ymins[j] = min(col_ymins[j], np.nanmin(mean_curve - sem_curve))
                    col_ymaxs[j] = max(col_ymaxs[j], np.nanmax(mean_curve + sem_curve))
                    drew_any = True

                if (null_curve is not None) and np.any(np.isfinite(null_curve)):
                    ax.plot(time, null_curve, linestyle='--', linewidth=2,
                            color=area_colors.get(area, 'black'), alpha=0.7, zorder=0)
                    col_ymins[j] = min(col_ymins[j], np.nanmin(null_curve))
                    col_ymaxs[j] = max(col_ymaxs[j], np.nanmax(null_curve))
                    drew_any = True

            ax.axvline(0, color='k', linestyle='--', linewidth=1) #stimOn
            ax.axvline(0.15, color='k', linestyle='--', linewidth=1) #stimOff

            ax.axvline(0.2, color='k', linestyle='--', linewidth=1) #stimOn
            ax.axvline(0.35, color='k', linestyle='--', linewidth=1) #stimOff

            ax.axvline(0.4, color='k', linestyle='--', linewidth=1) #stimOn
            ax.axvline(0.55, color='k', linestyle='--', linewidth=1) #stimOff

            ax.axvline(0.6, color='k', linestyle='--', linewidth=1) #stimOn
            ax.axvline(0.75, color='k', linestyle='--', linewidth=1) #stimOff

            ax.axvline(0.8, color='k', linestyle='--', linewidth=1) #stimOn
            ax.axvline(0.95, color='k', linestyle='--', linewidth=1) #stimOff

            ax.axvline(1, color='k', linestyle='--', linewidth=1) #stimOn
            ax.axvline(1.15, color='k', linestyle='--', linewidth=1) #stimOff
            
            ax.axvline(1.7, color='blue', linestyle='--', linewidth=1) #targetOn
            ax.axvline(1.8, color='blue', linestyle='--', linewidth=1) #targetOff

            ax.axvline(2.55, color='green', linestyle='--', linewidth=1) #fixptOff
            # ax.axvline(2.73, color='green', linestyle='--', linewidth=1) #responseDone

                         

            if not drew_any:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='gray')

            ax.tick_params(axis='both', which='major', labelsize=24, width=4, length=12)
            ax.tick_params(axis='both', which='minor', width=3, length=8)
            for spine in ax.spines.values():
                spine.set_linewidth(3)
            ax.set_xlim(time[0], time[-1])
            if i == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=26, fontweight='bold')

    for j in range(n_cols):
        if np.isfinite(col_ymins[j]) and np.isfinite(col_ymaxs[j]):
            for i in range(n_rows):
                axes[i, j].set_ylim(-0.2, 0.6)
        else:
            for i in range(n_rows):
                axes[i, j].set_ylim(-1.0, 1.0)

    for j, label in enumerate(column_labels):
        axes[0, j].set_title(label, fontsize=20, fontweight='bold', pad=20)

    fig.canvas.draw()
    row_y_positions = [np.mean([ax.get_position().y0, ax.get_position().y1]) for ax in axes[:, 0]]
    for y_pos, label in zip(row_y_positions, row_labels[:n_rows]):
        fig.text(0.005, y_pos, label, va='center', ha='left', fontsize=18, fontweight='bold', rotation='vertical')

    fig.text(0.002, 0.5, "Circcorr", va='center', ha='center',
             rotation='vertical', fontsize=28, fontweight='bold')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(areas),
                   fontsize=15, frameon=False, title='Area', title_fontsize=17)

    plt.tight_layout(rect=[0.04, 0.10, 1, 0.95])
    fig.savefig("/home/aarghavan/aslan/delsac-neural-decoding/results/pearsonr_with_shuffled_time_bin.svg", format='svg', bbox_inches='tight')
    plt.show()

with open('/home/aarghavan/aslan/delsac-neural-decoding/results/predicted_data.pkl', 'rb') as f: ## change it for your data
    all_predictions = pickle.load(f)


with open('/home/aarghavan/aslan/delsac-neural-decoding/results/suffled_trials.pkl', 'rb') as f: ## change it for your data
    all_shuffles = pickle.load(f)

with open('/home/aarghavan/aslan/delsac-neural-decoding/delsac_filtered_6_nMapStim.pkl', 'rb') as f: ## change it for your data
    data = pickle.load(f)

with open('/home/aarghavan/aslan/delsac-neural-decoding/results/processed_trials.pkl', 'rb') as f: ## change it for your data
    processed_trials = pickle.load(f)

data['trial'] = processed_trials


all_mean_scores, all_sem_scores, all_null_scores = aggregate_circcorrcoef_with_shuffles(
    all_predictions,
    all_shuffles,
    data,
    angle_pairs
)

bin_size = 0.1 ## change this according to your data
time = np.arange(-2.5, 3.5, bin_size)[:59] ## also change this

# areas = ['PFC', 'FEF', 'IT', 'MT', 'LIP', 'Parietal', 'V4']
areas = ['PFC', 'FEF', 'LIP']
# areas = ['IT', 'MT', 'Parietal', 'V4']
previous_variables = ['prevTargetAngle', 'prevRespAngle']
current_variables = ['targetAngle', 'respAngle']


plot_time_resolved_pearson(
    all_mean_scores,
    all_sem_scores,
    time,
    areas,
    previous_variables,
    current_variables,
    all_null_scores=all_null_scores
)

