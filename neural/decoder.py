from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from tqdm import trange,tqdm
import pickle
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import sys, os
# import pandas as pd
# from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut 
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
import scipy.signal
import itertools
from math import ceil
# import ssm
# from ssm.util import find_permutation
from sklearn.model_selection import KFold
import time 
from collections import defaultdict
import numpy as np
import pandas as pd
from astropy.stats import circcorrcoef


with open('/home/aarghavan/aslan/delsac-neural-decoding/delsac_all_bin_0.1_noNaN_aligned.pkl', 'rb') as f:
    data = pickle.load(f)

bin_size = 0.1
t_start = -2.5
t_end = 3.5
time = np.arange(t_start, t_end, bin_size)  # 59 bins, no need for [:59] slicing
stim_idx = (time > 0.5) & (time < 1.0)


num_sessions = len(data['spikecounts'])
print(f"Number of available sessions: {num_sessions}")

if num_sessions == 0:
    raise ValueError("No sessions found in data!")


def create_y(trial_data):
    y = {}

    y['targetX'] = trial_data['targetX'].values
    y['targetY'] = trial_data['targetY'].values
    y['respX'] = trial_data['respX'].values
    y['respY'] = trial_data['respY'].values

    y['prevTargetX'] = np.roll(y['targetX'], 1, axis=0)
    y['prevTargetY'] = np.roll(y['targetY'], 1, axis=0)
    y['prevRespX'] = np.roll(y['respX'], 1, axis=0)
    y['prevRespY'] = np.roll(y['respY'], 1, axis=0)
    
    cols_all = [
        'targetX','targetY', 'respX', 'respY', 'prevTargetX', 'prevTargetY', 'prevRespX', 'prevRespY'
    ]

    numeric_cols = trial_data.select_dtypes(include=[np.number]).columns
    trial_data[numeric_cols] = trial_data[numeric_cols].replace([np.inf, -np.inf], np.nan)
    trial_data[cols_all] = trial_data[cols_all].fillna(0) 

    return y


print("\nVerifying session alignment:")
for s, sc in enumerate(data['spikecounts']):
    n_spikes = sc.shape[1]
    n_units  = len(data['unit'][s])
    print(f"Session {s}: spikecounts neurons = {n_spikes}, unit entries = {n_units}")
    if n_spikes != n_units:
        raise ValueError(f"❌ Mismatch in session {s}: spikecounts={n_spikes} vs units={n_units}")
print("✅ All sessions verified.\n")


required = {'targetX', 'targetY', 'respX', 'respY' }

for i, df in enumerate(data['trial']):
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        print(f"Session {i}: skipped (missing columns: {missing})")
        continue

    df = df.copy()

    def norm(x):
        return 2 * (x - x.min()) / (x.max() - x.min()) - 1      # [-1, 1]

    df['targetX'] = norm(df['targetX'])
    df['targetY'] = norm(df['targetY'])
    df['respX']   = norm(df['respX'])
    df['respY']   = norm(df['respY'])
    df['prevTargetX'] = df['targetX'].shift(1)
    df['prevTargetY'] = df['targetY'].shift(1)
    df['prevRespX']   = df['respX'].shift(1)
    df['prevRespY']   = df['respY'].shift(1)


    # Save back
    data['trial'][i] = df

print("✨ Cleaning + transformations complete.")




areas = ['PFC', 'FEF', 'IT', 'MT', 'LIP', 'Parietal', 'V4']
previous_variables = ['prevTargetX', 'prevTargetY', 'prevRespX','prevRespY']
current_variables = ['targetX', 'targetY', 'respX',  'respY']
variables_to_analyze = previous_variables + current_variables



def smoothing(X, bin_size=0.1, K=1.0, width=2.0): 

    """
    Applies exponential smoothing to spike data (one session).

    Parameters:
    X : np.array
        Spike data (trials x neurons x time bins)
    bin_size : float
        Width of each time bin (in seconds)
    K : float
        Shape parameter for exponential decay
    width : float
        Controls smoothing extent (in seconds)

    Returns:
    X_smoothed : np.array
        Smoothed spike data
    """
    bin_w = int(ceil(width / bin_size))
    win = scipy.signal.windows.exponential(2 * bin_w + 1, tau=bin_w / (2 * K))
    win[:bin_w] = 0
    win /= win.sum() * bin_size  # Normalize for area under the curve

    new_data = np.zeros_like(X)
    convol_fun = lambda x: np.convolve(x, win, mode='same')

    for c, n in itertools.product(range(X.shape[0]), range(X.shape[1])):
        new_data[c, n, :] = convol_fun(X[c, n, :])

    return new_data



def load_smoothed_centered_data(data, centered_file_path, smoothed_file_path, bin_size=0.1):
    """
    Load smoothed and centered spikecounts and unit metadata from disk.

    """

    # === Step 1: Try loading centered data (spikecounts + unit) ===
    if os.path.exists(centered_file_path):
        print('Centered + smoothed data already exists.')
        with open(centered_file_path, 'rb') as f:
            spike_unit_package = pickle.load(f)

        if isinstance(spike_unit_package, dict) and 'spikecounts' in spike_unit_package:
            data['spikecounts'] = spike_unit_package['spikecounts']
            data['unit'] = spike_unit_package['unit']
            print('  Loaded centered + smoothed spikecounts and unit metadata.')
        else:
            raise ValueError("Invalid format in centered file — expected a dict with 'spikecounts' and 'unit'.")

    # === Step 2: If centered data doesn't exist, check for smoothed data ===
    else:
        if os.path.exists(smoothed_file_path):
            print('Smoothed data found (not centered).')
            with open(smoothed_file_path, 'rb') as f:
                data['spikecounts'] = pickle.load(f)
            print('  Loaded smoothed spikecounts.')
        else:
            print('No smoothed data found — computing smoothing...')
            data['spikecounts'] = [
                smoothing(np.array(session_data), bin_size=bin_size, width=1.5, K=2)
                for session_data in tqdm(data['spikecounts'], desc='Smoothing Data')
            ]
            with open(smoothed_file_path, 'wb') as f:
                pickle.dump(data['spikecounts'], f)
            print('  Saved smoothed data.')

        # === Step 3: Now center the smoothed data ===
        print('Centering smoothed spikecounts...')
        for s, d in enumerate(tqdm(data['spikecounts'], desc='Centering Data')):
            # Reshape to neuron × time
            d_s = np.einsum('ijk -> jik', d).reshape(d.shape[1], -1)

            # Filter neurons with non-zero standard deviation
            stdev = np.std(d_s, axis=1)
            idx = stdev > 0
            stdev = stdev[idx]
            mmean = np.mean(d_s[idx], axis=1)

            # Filter both unit and spikecounts data
            data['unit'][s] = data['unit'][s].loc[idx].reset_index(drop=True)
            data['spikecounts'][s] = (d[:, idx] - mmean[None, :, None]) / stdev[None, :, None]

            # Final safety check
            assert data['spikecounts'][s].shape[1] == len(data['unit'][s]), f"Mismatch in session {s}"

        # === Step 4: Save the centered data (both spikecounts and unit metadata) ===
        with open(centered_file_path, 'wb') as f:
            pickle.dump({'spikecounts': data['spikecounts'], 'unit': data['unit']}, f)
        print('  Saved centered + smoothed spikecounts and unit metadata.')

    return data



centered_file_path = '/home/aarghavan/aslan/delsac-neural-decoding/results/centered.pkl'
smoothed_file_path = '/home/aarghavan/aslan/delsac-neural-decoding/results/smoothed.pkl'

data = load_smoothed_centered_data(data, centered_file_path, smoothed_file_path)





def decode_one_session_timebin(
    session, data, area, variables_to_analyze,
    compute_null=True, n_shuffles=5, cv_real=5, cv_null=3, random_state=0
):
    """
    RETURNS (4 items):
        session, session_preds, trial_data, session_shuffles

    - session_preds[v]['predictions'] -> (n_trials, n_timebins)
    - session_shuffles[v] -> (n_shuffles, n_trials, n_timebins)  (for Pearson r null)
    """
    print(f"\n--- Decoding session {session} for area {area} ---")

    area_idx = data['unit'][session]['area'].values == area
    if np.sum(area_idx) < 10:
        print(f"  Skipping session {session}: too few neurons ({np.sum(area_idx)})")
        return session, None, None, None

    spike_data = data['spikecounts'][session][:, area_idx, :]  # (trials, neurons, time)
    trial_data = data['trial'][session]
    y_all = create_y(trial_data)
    for key, values in y_all.items():
        trial_data[key] = values

    session_preds = {}
    session_shuffles = {} if compute_null else None

    n_trials, _, n_timebins = spike_data.shape

    for v in variables_to_analyze:
        print(f"    Decoding variable: {v}")
        target = np.asarray(y_all[v])

        if np.unique(target).size < 2:
            print(f"      Skipping {v}: only one class present")
            continue

        preds_timebin = np.full((n_trials, n_timebins), np.nan, dtype=float)
        shuf_store = (np.full((n_shuffles, n_trials, n_timebins), np.nan, dtype=float)
                      if compute_null else None)

        # Prepare CV objects once per variable # this a simple version  with cross validation with k-fold, can be switched with LOO (more computationally expensive)
        # kf_real = KFold(n_splits=cv_real, shuffle=True, random_state=random_state)
        # kf_null = KFold(n_splits=cv_null, shuffle=True, random_state=random_state)

        loo_real = LeaveOneOut()
        loo_null = LeaveOneOut() 
        
        clf = Pipeline([
            ('standardize', StandardScaler()),
            #('PCA', PCA(n_components=0.9)),
            #LinearSVC(dual='auto', max_iter=2000, tol=1e-2)
            ('reg', RidgeCV(alphas=np.linspace(0.001,1.5,10))) 
        ])

        for t in range(n_timebins):
            Xt = spike_data[:, :, t]
            if np.all(Xt == 0):
                continue

            # ---- REAL predictions (cv_real) ---- # it means the acutual predictions
            real_preds = np.full(n_trials, np.nan, dtype=float)
            for tr_idx, te_idx in loo_real.split(Xt):
                clf.fit(Xt[tr_idx], target[tr_idx])
                real_preds[te_idx] = clf.predict(Xt[te_idx])
            preds_timebin[:, t] = real_preds

            # ---- NULL predictions (cv_null) ---- # we are doing the permutaion test
            if compute_null:
                for s in range(n_shuffles):
                    y_shuf = np.random.permutation(target)
                    shuf_preds = np.full(n_trials, np.nan, dtype=float)
                    for tr_idx, te_idx in loo_null.split(Xt):
                        clf.fit(Xt[tr_idx], y_shuf[tr_idx])
                        shuf_preds[te_idx] = clf.predict(Xt[te_idx])
                    shuf_store[s, :, t] = shuf_preds

        session_preds[v] = {'predictions': preds_timebin}
        if compute_null:
            session_shuffles[v] = shuf_store

    print(f"  ✓ Finished decoding session {session} for area {area}")
    return session, session_preds, trial_data, session_shuffles




def decode_area_loo_parallel_timebin(
    data, sessions, area, variables_to_analyze,
    compute_null=False, n_shuffles=100,
    cv_real=5, cv_null=3, random_state=0   # <-- add random_state here
):
    """
    Returns:
        data, predicted_results, shuffled_results
    predicted_results[area][session][var]['predictions'] -> (n_trials, n_time)
    shuffled_results[area][session][var] -> (n_shuffles, n_trials, n_time)
    """
    print(f"\n>>> Starting decoding for area: {area} (n_sessions={len(sessions)})")

    results = Parallel(n_jobs=-1)(
        delayed(decode_one_session_timebin)(
            session, data, area, variables_to_analyze,
            compute_null=compute_null, n_shuffles=n_shuffles,
            cv_real=cv_real, cv_null=cv_null, random_state=random_state
        )
        for session in sessions
    )

    predicted_results = defaultdict(dict)
    shuffled_results  = defaultdict(dict)

    for res in results:
        if res is None:
            continue
        if len(res) == 4:
            session, preds, updated_trial_data, shuf_dict = res
        elif len(res) == 3:
            session, preds, updated_trial_data = res
            shuf_dict = None
        else:
            continue

        if preds is not None:
            predicted_results[area][session] = preds
            data['trial'][session] = updated_trial_data
        if shuf_dict is not None:
            shuffled_results[area][session] = shuf_dict

    return data, predicted_results, shuffled_results



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
    row_labels = ['targetAng', 'respAng']
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

            # ax.axvline(0, color='k', linestyle='--', linewidth=1) #stimOn
            # ax.axvline(0.15, color='k', linestyle='--', linewidth=1) #stimOff

            # ax.axvline(0.2, color='k', linestyle='--', linewidth=1) #stimOn
            # ax.axvline(0.35, color='k', linestyle='--', linewidth=1) #stimOff

            # ax.axvline(0.4, color='k', linestyle='--', linewidth=1) #stimOn
            # ax.axvline(0.55, color='k', linestyle='--', linewidth=1) #stimOff

            # ax.axvline(0.6, color='k', linestyle='--', linewidth=1) #stimOn
            # ax.axvline(0.75, color='k', linestyle='--', linewidth=1) #stimOff

            # ax.axvline(0.8, color='k', linestyle='--', linewidth=1) #stimOn
            # ax.axvline(0.95, color='k', linestyle='--', linewidth=1) #stimOff

            # ax.axvline(1, color='k', linestyle='--', linewidth=1) #stimOn
            # ax.axvline(1.15, color='k', linestyle='--', linewidth=1) #stimOff
            
            # ax.axvline(1.7, color='blue', linestyle='--', linewidth=1) #targetOn
            # ax.axvline(1.8, color='blue', linestyle='--', linewidth=1) #targetOff

            # ax.axvline(2.55, color='green', linestyle='--', linewidth=1) #fixptOff
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
                axes[i, j].set_ylim(-0.2, 0.5)
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



################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

def calculate_prediction_errors(all_predictions, data, angle_pairs):
    """
    Calculate prediction errors for both Cartesian and angular variables.
    
    Returns:
        errors_cartesian: dict[area][session][var] -> (n_trials, n_timebins)
        errors_angular: dict[area][session][angle_name] -> (n_trials, n_timebins)
    """
    errors_cartesian = {}
    errors_angular = {}
    
    for area in all_predictions.keys():
        errors_cartesian[area] = {}
        errors_angular[area] = {}
        
        for session in all_predictions[area].keys():
            preds_dict = all_predictions[area][session]
            trial_df = data['trial'][session]
            
            errors_cartesian[area][session] = {}
            errors_angular[area][session] = {}
            
            # --- Cartesian errors (X, Y) ---
            for var in preds_dict.keys():
                predictions = np.asarray(preds_dict[var]['predictions'])  # (n_trials, n_time)
                targets = np.asarray(trial_df[var])[:, None]  # (n_trials, 1)
                
                # Signed error
                error = predictions - targets  # (n_trials, n_time)
                errors_cartesian[area][session][var] = error
            
            # --- Angular errors ---
            for angle_name, (varX, varY) in angle_pairs.items():
                if varX not in preds_dict or varY not in preds_dict:
                    continue
                
                # Predictions
                predsX = np.asarray(preds_dict[varX]['predictions'])
                predsY = np.asarray(preds_dict[varY]['predictions'])
                pred_angle = np.arctan2(predsY, predsX)  # (n_trials, n_time)
                
                # Targets
                trueX = np.asarray(trial_df[varX])
                trueY = np.asarray(trial_df[varY])
                true_angle = np.arctan2(trueY, trueX)[:, None]  # (n_trials, 1)
                
                # Circular error (in radians, range: -π to π)
                angular_error = np.arctan2(
                    np.sin(true_angle - pred_angle),
                    np.cos(true_angle - pred_angle)
                )
                errors_angular[area][session][angle_name] = angular_error
    
    return errors_cartesian, errors_angular


def compute_error_statistics(errors_dict, time):
    """
    Compute mean ± SEM of signed errors across sessions.
    
    Returns:
        mean_errors: dict[area][var] -> (n_timebins,)
        sem_errors: dict[area][var] -> (n_timebins,)
    """
    mean_errors = {}
    sem_errors = {}
    
    for area in errors_dict.keys():
        mean_errors[area] = {}
        sem_errors[area] = {}
        
        # Collect errors from all sessions for this area
        per_var_sessions = {}
        
        for session in errors_dict[area].keys():
            for var in errors_dict[area][session].keys():
                if var not in per_var_sessions:
                    per_var_sessions[var] = []
                
                error = errors_dict[area][session][var]
                # Compute mean signed error per trial, then average over trials
                error_per_time = np.nanmean(error, axis=0)  # (n_time,)
                per_var_sessions[var].append(error_per_time)
        
        # Average across sessions
        for var, session_list in per_var_sessions.items():
            stacked = np.stack(session_list, axis=0)  # (n_sessions, n_time)
            mean_errors[area][var] = np.nanmean(stacked, axis=0)
            n_eff = np.sum(~np.isnan(stacked), axis=0)
            sem_errors[area][var] = np.nanstd(stacked, axis=0) / np.sqrt(n_eff)
    
    return mean_errors, sem_errors

def plot_prediction_errors(mean_errors, sem_errors, time, areas, 
                           angle_pairs, save_path=None):
    """
    Plot time-resolved prediction errors for angular variables.
    """
    n_angles = len(angle_pairs)
    fig, axes = plt.subplots(n_angles, 1, figsize=(12, 4 * n_angles), sharex=True)
    if n_angles == 1:
        axes = [axes]
    
    area_colors = {'PFC':'b', 'FEF':'r', 'LIP':'g', 'Parietal':'y', 
                   'IT':'c', 'MT':'m', 'V4':'orange'}
    
    for i, angle_name in enumerate(angle_pairs.keys()):
        ax = axes[i]
        
        for area in areas:
            if angle_name in mean_errors[area]:
                mean = mean_errors[area][angle_name]
                sem = sem_errors[area][angle_name]
                
                ax.plot(time, mean, label=area, linewidth=2.5,
                       color=area_colors.get(area, 'black'))
                ax.fill_between(time, mean - sem, mean + sem,
                               alpha=0.3, color=area_colors.get(area, 'black'))
        
        # Mark stimulus events
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(1.7, color='blue', linestyle='--', alpha=0.5, label='Target On')
        ax.axvline(2.55, color='green', linestyle='--', alpha=0.5, label='Response')
        
        ax.set_ylabel(f'{angle_name}\nError (rad)', fontsize=14, fontweight='bold')
        ax.set_title(f'Neural Decoding Error: {angle_name}', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


import time #  maybe change the name

num_sessions = len(data['spikecounts'])

print(f"Number of available sessions: {num_sessions}")

if num_sessions == 0:
    raise ValueError("No sessions found in data!")

# Only decode the first 3 sessions
sessions = list(range(min(100, num_sessions)))
# sessions = list(range(num_sessions))

# === Define Parameters ===
areas = ['PFC', 'FEF', 'IT', 'MT', 'LIP', 'Parietal', 'V4']
# areas = ['PFC', 'FEF', 'LIP']
# areas = ['PFC']
# areas = ['IT', 'MT', 'Parietal', 'V4']
previous_variables = ['prevTargetX', 'prevTargetY', 'prevRespX', 'prevRespY']
current_variables = ['targetX', 'targetY', 'respX',  'respY']
variables_to_analyze = previous_variables + current_variables



compute_null = True
n_shuffles = 2 #at the begging try this, later increase
cv_real=5 # as it's the actual prediction
cv_null=3 # as it's just the permuation test, it can be lowert o be xomputtaionlly more cheap

all_predictions = {}
all_shuffles    = {}

for area in areas:
    print(f"\n=== Checking sessions for area: {area} ===")
    valid_sessions = []

    for session in sessions:
        area_idx = data['unit'][session]['area'].values == area
        num_neurons = np.sum(area_idx)

        if num_neurons >= 5:
            print(f"  ✓ Session {session} → {num_neurons} neurons")
            valid_sessions.append(session)
        else:
            print(f"  ✗ Session {session} skipped → only {num_neurons} neurons")

    if valid_sessions:
        print(f"==> Decoding {len(valid_sessions)} sessions for area {area}")
        start_time = time.time()
        data, area_preds, area_shufs = decode_area_loo_parallel_timebin(
            data, valid_sessions, area, variables_to_analyze,
            compute_null=compute_null, n_shuffles=n_shuffles,
            cv_real=cv_real, cv_null=cv_null, random_state=0
        )

        all_predictions.update(area_preds)
        all_shuffles.update(area_shufs)

        elapsed = time.time() - start_time
        print(f"✓ Finished decoding area {area} in {elapsed:.2f}s")
    else:
        print(f"No sessions to decode for area {area}.")



# Save updated trial data (optional, if you added prediction info into it)
with open('/home/aarghavan/aslan/delsac-neural-decoding/results/processed_trials.pkl', 'wb') as f:
    pickle.dump(data['trial'], f) # optional, I save it to keep track but you can skip this step

with open('/home/aarghavan/aslan/delsac-neural-decoding/results/suffled_trials.pkl', 'wb') as f:
    pickle.dump(all_shuffles, f)  # the shuffles and actual predictions are stored in spearte files, as it's easier for me to load and plot them

# ✅ Save full prediction results, including null distributions
with open('/home/aarghavan/aslan/delsac-neural-decoding/results/predicted_data.pkl', 'wb') as f:
    pickle.dump(all_predictions, f)



all_mean_scores, all_sem_scores, all_null_scores = aggregate_circcorrcoef_with_shuffles(
    all_predictions,
    all_shuffles,
    data,
    angle_pairs
)

# Save per-session correlation curves (before averaging)
per_session_curves = {}
for area in all_predictions.keys():
    per_session_curves[area] = {}
    for session in all_predictions[area].keys():
        preds_dict = all_predictions[area][session]
        trial_df = data['trial'][session]
        per_session_curves[area][session] = {}
        
        for angle_name, (varX, varY) in angle_pairs.items():
            if varX not in preds_dict or varY not in preds_dict:
                continue
            predsX = np.asarray(preds_dict[varX]['predictions'])
            predsY = np.asarray(preds_dict[varY]['predictions'])
            pred_angle = np.arctan2(predsY, predsX)
            
            trueX = np.asarray(trial_df[varX])
            trueY = np.asarray(trial_df[varY])
            true_angle = np.arctan2(trueY, trueX)
            valid = ~np.isnan(true_angle)
            
            n_time = pred_angle.shape[1]
            curve = np.full(n_time, np.nan)
            for t in range(n_time):
                p = pred_angle[valid, t]
                y = true_angle[valid]
                if not np.all(np.isnan(p)):
                    curve[t] = circcorrcoef(p, y)
            
            per_session_curves[area][session][angle_name] = curve

with open('/home/aarghavan/aslan/delsac-neural-decoding/results/per_session_circcorr.pkl', 'wb') as f:
    pickle.dump(per_session_curves, f)
print("✓ Saved per-session circular correlation curves.")

bin_size = 0.1 ## change this according to your data
time = np.arange(-2.5, 3.5, bin_size)[:59] ## also change this
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

#################################################################################################################
#################################################################################################################
#################################################################################################################


errors_cartesian, errors_angular = calculate_prediction_errors(
    all_predictions, data, angle_pairs
)

mean_angular_errors, sem_angular_errors = compute_error_statistics(
    errors_angular, time
)

# === SAVE neural errors as CSV ===
import pandas as pd

# Flatten errors_angular into a dataframe format
rows = []
for area in errors_angular.keys():
    for session in errors_angular[area].keys():
        for angle_name in errors_angular[area][session].keys():
            error_array = errors_angular[area][session][angle_name]  # (n_trials, n_time)
            
            # Get original trial metadata
            trial_df = data['trial'][session]
            
            for trial_idx in range(error_array.shape[0]):
                # Extract original session and trial from behavioral data
                orig_session = trial_df.iloc[trial_idx]['session'] if 'session' in trial_df.columns else None
                orig_trial = trial_df.iloc[trial_idx]['trial'] if 'trial' in trial_df.columns else None
                
                for time_idx in range(error_array.shape[1]):
                    rows.append({
                        'area': area,
                        'session': session,
                        'original_session': orig_session,
                        'original_trial': orig_trial,
                        'angle_name': angle_name,
                        'trial': trial_idx,
                        'time_idx': time_idx,
                        'time': time[time_idx],
                        'error': error_array[trial_idx, time_idx]
                    })

errors_df = pd.DataFrame(rows)
errors_df.to_csv('/home/aarghavan/aslan/delsac-neural-decoding/results/errors_angular.csv', index=False)

print("✓ Saved neural errors to CSV")
print(f"  Shape: {errors_df.shape}")
print(f"  Columns: {errors_df.columns.tolist()}")

plot_prediction_errors(
    mean_angular_errors,
    sem_angular_errors,
    time,
    areas,
    angle_pairs,
    save_path="/home/aarghavan/aslan/delsac-neural-decoding/results/prediction_errors.svg"
)
#################################################################################################################
#################################################################################################################
#################################################################################################################


