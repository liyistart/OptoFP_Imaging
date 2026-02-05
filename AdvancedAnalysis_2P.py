import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from scipy.stats import zscore
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.linalg import svd

################################################################################################################
# Dimensionality Reduction functions
#################################################################################################################

def compute_pca_projections(full_mat_info, n_pca_components=8):
    """
    Computes PCA projections and total variance explained across all trials.
    
    Parameters
    ----------
    full_mat_info : ndarray
        Neural or behavioral data (n_trials x n_frames x n_features)
    n_pca_components : int
        Number of PCA components to keep
    
    Returns
    -------
    pca_projections : ndarray
        PCA projections (n_trials x n_frames x n_pca_components)
    total_var_explained : float
        Fraction of total variance explained by the retained PCA components
    """
    n_trials, n_frames, n_features = full_mat_info.shape

    # Z-score normalization per feature
    norm_full = np.zeros_like(full_mat_info)
    for f in range(n_features):
        norm_full[:, :, f] = zscore(full_mat_info[:, :, f], axis=None, ddof=1)

    # PCA on trial-time flattened data
    input_matrix = norm_full.reshape(-1, n_features)
    pca = PCA(n_components=n_pca_components).fit(input_matrix)
    pca_projections = pca.transform(input_matrix).reshape(n_trials, n_frames, n_pca_components)

    # Reconstruct the data from PCA projections
    reconstructed = pca.inverse_transform(pca_projections.reshape(-1, n_pca_components))
    reconstructed = reconstructed.reshape(n_trials, n_frames, n_features)

    # Compute total variance explained
    total_var = np.nanvar(norm_full)
    explained_var = np.nanvar(reconstructed)
    total_var_explained = explained_var / total_var  # Fraction of variance explained

    return pca_projections, total_var_explained


def compute_lda_projection(pca_projections, dev_on_frames, trial_sess_id=None, n_comps=50, fp_frame_num=300, cv_num=10):
    """
    Session-aware LDA projection with full temporal trajectories.
    Averages projections across cv_num cross-validation runs.
    Uses random 50% of trials for training and sets their projections to NaN.
    """
    # Get dimensions
    n_trials, n_frames, n_components = pca_projections.shape
    
    # Select components based on variance
    variance = np.var(pca_projections, axis=(0,1))
    var_ratio = variance / np.nansum(variance)
    high_var_comps = np.where(var_ratio >= 0.01)[0]
    n_comps = min(n_comps, high_var_comps.size)
    print(f"Using first {n_comps} components out of {n_components} based on variance threshold")
    
    # Initialize arrays for CV results
    all_projections = np.full((cv_num, n_trials, n_frames), np.nan)
    
    for cv_idx in range(cv_num):
        projections = np.full((n_trials, n_frames), np.nan)
        lda_weights = {}
        ldas = {}
        
        for session in np.unique(trial_sess_id):
            # Get session trials
            sess_mask = (trial_sess_id == session)
            sess_idx = np.where(sess_mask)[0]
            reach_mask = np.zeros(len(dev_on_frames), dtype=bool)
            reach_mask[:140] = True
            
            # Find perturbed and control trials
            perturbed_mask = ~np.isnan(dev_on_frames)
            perturbed_trials = np.where(sess_mask & perturbed_mask & reach_mask)[0]
            control_trials = np.where(sess_mask & ~perturbed_mask & reach_mask)[0]
            
            # Get median perturbation onset
            median_onset = int(np.nanmean(dev_on_frames[perturbed_trials]))
            
            # Split trials with different random state for each CV iteration
            random_state = 42 + cv_idx
            perturbed_train, perturbed_test = train_test_split(perturbed_trials, test_size=0.5, random_state=random_state)
            control_train, control_test = train_test_split(control_trials, test_size=0.5, random_state=random_state)
            
            training_trials = np.concatenate([perturbed_train, control_train])
            test_trials = np.concatenate([perturbed_test, control_test])
            
            # Extract training data using peaks
            X_train = []
            y_train = []
            
            for i in training_trials:
                try:
                    start = int(dev_on_frames[i]) if i in perturbed_trials else median_onset
                    if start + fp_frame_num <= n_frames:
                        window = pca_projections[i, start:start+fp_frame_num, :n_comps]
                        
                        # # Take peak across the window
                        # peak_idx = np.argmax(np.abs(window), axis=0)
                        # peak_values = window[peak_idx, np.arange(n_comps)]
                        # X_train.append(peak_values)

                        # Take mean across the window instead of peak
                        mean_values = np.mean(window, axis=0)  # shape: (n_comps,)
                        X_train.append(mean_values)

                        y_train.append(i in perturbed_trials)
                except Exception as e:
                    print(f"Error processing trial {i}: {str(e)}")
                    continue

            if len(X_train) < 2:
                print(f"Skipping session {session}: not enough valid trials")
                continue
                
            # Convert lists to arrays
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Fit LDA
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train, y_train)
            
            # Normalize weights
            weights = lda.coef_[0]
            weights = weights / np.linalg.norm(weights)
            
            # Store results
            ldas[session] = lda
            lda_weights[session] = weights
            
            # Project test data
            for t in sess_idx:
                if t in training_trials:
                    continue
                for f in range(n_frames):
                    frame_data = pca_projections[t, f, :n_comps]
                    projections[t, f] = np.dot(frame_data, weights)
        
        # Store projections for this CV iteration
        all_projections[cv_idx] = projections
    
    # Average projections across CV runs
    avg_projections = np.nanmean(all_projections, axis=0)
    
    return avg_projections, lda_weights, ldas

def compute_pls_projections(full_mat_info, dev_on_frames, n_pls_components=20):
    """
    Compute PLS projections for 3D behavioral data.

    Parameters
    ----------
    full_mat_info : np.ndarray
        Shape (n_trials, n_frames, n_features)
    dev_on_frames : np.ndarray
        Perturbation onset frames (NaN for controls)
    n_pls_components : int
        Number of PLS components

    Returns
    -------
    pls_projections : np.ndarray
        Shape (n_trials, n_frames, n_pls_components)
    reference : np.ndarray
        Mean PLS trajectory for control trials (n_frames, n_pls_components)
    reconstruction_error : np.ndarray
        MSE per trial
    """
    n_trials, n_frames, n_features = full_mat_info.shape

    # Z-score normalization per feature
    norm_full = np.zeros_like(full_mat_info)
    for f in range(n_features):
        norm_full[:, :, f] = zscore(full_mat_info[:, :, f], axis=None, ddof=1)

    # Prepare data for PLS
    X_full = norm_full.reshape(-1, n_features)
    # Binary label for each frame: 1 if perturbed, 0 if control
    y_full = np.repeat(~np.isnan(dev_on_frames), n_frames).astype(int).reshape(-1, 1)

    # Fit PLS
    pls = PLSRegression(n_components=n_pls_components)
    pls.fit(X_full, y_full)
    pls_projections = pls.transform(X_full).reshape(n_trials, n_frames, n_pls_components)

    # Reconstruct the data from PLS projections
    reconstructed = pls.inverse_transform(pls_projections.reshape(-1, n_pls_components))
    reconstructed = reconstructed.reshape(n_trials, n_frames, n_features)

    # Calculate reconstruction error (mean squared error) per trial
    reconstruction_error = np.nanmean((norm_full - reconstructed) ** 2, axis=(1, 2))  # MSE per trial

    # Reference trajectory: mean PLS projection for control trials
    control_mask = np.isnan(dev_on_frames)
    if np.nansum(control_mask) == 0:
        raise ValueError("No control trials found for reference trajectory")
    control_pls_projections = pls_projections[control_mask]
    reference = np.nanmean(control_pls_projections, axis=0)

    return pls_projections, reference, reconstruction_error

################################################################################################################
# Plane fitting function (modified)
################################################################################################################

def fit_trk_plane(traj, orient_z_positive=True):
    """
    Fit a plane (z = ax + by + c) to a 3D trajectory using linear regression.

    Parameters
    ----------
    traj : np.ndarray
        Array of shape (n_samples, 3), where columns are (x, y, z).
    orient_z_positive : bool
        If True, force the normal vector to have a positive z-component.

    Returns
    -------
    plane_normal : np.ndarray
        Unit normal vector to the plane
    r2 : float
        Coefficient of determination (R² score)
    reg : LinearRegression
        Trained regression model
    plane_offset : float
        Offset 'd' in the implicit plane equation n.x + d = 0, 
        suitable for signed distance calculations.
    """
    if traj.shape[1] != 3:
        raise ValueError("Input trajectory must have shape (n_samples, 3)")

    X = traj[:, :2]   # x, y
    z = traj[:, 2]    # z

    reg = LinearRegression().fit(X, z)
    a, b = reg.coef_
    c = reg.intercept_

    # Normal vector of the plane in implicit form: a*x + b*y - z + c = 0
    plane_normal = np.array([a, b, -1.0])
    plane_normal /= np.linalg.norm(plane_normal)

    # Adjust normal and intercept to enforce positive z direction
    if orient_z_positive and plane_normal[2] < 0:
        plane_normal = -plane_normal
        c = -c  # flip intercept consistently

    # plane_offset can be used for signed distance computations
    plane_offset = c  # so distance = proj @ plane_normal + plane_offset

    r2 = reg.score(X, z)
    return plane_normal, r2, reg, plane_offset

# === Compute RMSE for each fit ===
def compute_rmse(traj, reg):
    X = traj[:, :2]
    z = traj[:, 2]
    z_pred = reg.predict(X)
    residuals = z - z_pred
    return np.sqrt(np.mean(residuals**2))

# === Projection of trajectories onto control plane ===
def project_onto_plane(points, normal, d):
    """
    Project points onto a plane defined by its normal vector and offset d.
    Plane equation: ax + by + cz + d = 0
    """
    normal = normal / np.linalg.norm(normal)
    distances = (points @ normal[:3] + d)
    projections = points - np.outer(distances, normal)
    return projections


################################################################################################################
# Condition separation functions
################################################################################################################
def compute_lda_and_residual_projection(pca_projections, dev_on_frames, trial_sess_id=None, n_comps=50, fp_frame_num=300, cv_num=10, reach_trials_per_session=140):
    """
    Session-aware LDA projection with full temporal trajectories.
    Averages projections across cv_num cross-validation runs.
    Uses random 50% of trials for training and sets their projections to NaN.

    Additionally computes a CV-averaged 1D residual projection by selecting
    a unit vector from the subspace orthogonal to the fitted LDA weight
    (obtained via SVD of the weight vector) and projecting activity onto it.
    Returns:
        avg_projections (n_trials x n_frames),
        avg_residual_projections (n_trials x n_frames),
        lda_weights_all (dict per session),
        lda_models_all (dict per session)
    """
    n_trials, n_frames, n_components = pca_projections.shape

    # Select components based on variance (keep at most n_comps)
    variance = np.var(pca_projections, axis=(0,1))
    var_ratio = variance / np.nansum(variance)
    high_var_comps = np.where(var_ratio >= 0.01)[0]
    n_comps = min(n_comps, high_var_comps.size, n_components)
    print(f"Using first {n_comps} components out of {n_components} based on variance threshold")

    # Initialize CV storage
    all_projections = np.full((cv_num, n_trials, n_frames), np.nan)
    all_residuals = np.full((cv_num, n_trials, n_frames), np.nan)

    lda_weights_all = {}
    lda_models_all = {}
    residual_bases_all = {}

    if trial_sess_id is None:
        trial_sess_id = np.zeros(n_trials, dtype=int)

    for cv_idx in range(cv_num):
        projections = np.full((n_trials, n_frames), np.nan)
        residual_projections = np.full((n_trials, n_frames), np.nan)

        for session in np.unique(trial_sess_id):
            sess_mask = (trial_sess_id == session)
            sess_idx = np.where(sess_mask)[0]
            if sess_idx.size == 0:
                continue
            reach_mask = np.zeros(len(dev_on_frames), dtype=bool)
            reach_mask[:reach_trials_per_session] = True

            perturbed_mask = ~np.isnan(dev_on_frames)
            perturbed_trials = np.where(sess_mask & perturbed_mask & reach_mask)[0]
            control_trials = np.where(sess_mask & ~perturbed_mask & reach_mask)[0]

            if perturbed_trials.size == 0 or control_trials.size == 0:
                continue

            # median onset (fallback to 0 if nan)
            try:
                median_onset = int(np.nanmean(dev_on_frames[perturbed_trials]))
            except Exception:
                median_onset = 0

            # CV split (different random_state per cv)
            random_state = 42 + cv_idx
            try:
                perturbed_train, perturbed_test = train_test_split(perturbed_trials, test_size=0.5, random_state=random_state)
                control_train, control_test = train_test_split(control_trials, test_size=0.5, random_state=random_state)
            except Exception:
                continue

            training_trials = np.concatenate([perturbed_train, control_train])
            test_trials = np.concatenate([perturbed_test, control_test])
            if training_trials.size < 2:
                continue

            # Build training matrix using mean across the fixed window
            X_train = []
            y_train = []
            for i in training_trials:
                try:
                    start = int(dev_on_frames[i]) if i in perturbed_trials else median_onset
                    if start + fp_frame_num <= n_frames:
                        window = pca_projections[i, start:start+fp_frame_num, :n_comps]
                        mean_values = np.mean(window, axis=0)
                        X_train.append(mean_values)
                        y_train.append(i in perturbed_trials)
                except Exception:
                    continue

            if len(X_train) < 2:
                continue

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # Fit LDA
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train, y_train)

            # Normalize weight
            weights = lda.coef_[0].astype(float)
            w_norm = np.linalg.norm(weights)
            if w_norm == 0:
                continue
            weights = weights / w_norm

            # Save last-seen LDA per session (overwritten across CVs)
            lda_models_all[session] = lda
            lda_weights_all[session] = weights

            # Compute one orthonormal residual basis vector orthogonal to weights:
            # Use SVD on the 1 x n_comps weight to obtain orthonormal complement.
            # Vt[0] aligns with weights (up to sign), Vt[1]...Vt[n_comps-1] span orthogonal subspace.
            try:
                _, _, Vt = svd(weights.reshape(1, -1), full_matrices=True)
                if Vt.shape[0] > 1:
                    u_res = Vt[1].astype(float)
                    u_res = u_res / np.linalg.norm(u_res)
                else:
                    # fallback: orthogonalize first basis vector
                    tmp = np.zeros(n_comps); tmp[0] = 1.0
                    tmp = tmp - (tmp @ weights) * weights
                    if np.linalg.norm(tmp) == 0:
                        u_res = np.zeros_like(weights)
                    else:
                        u_res = tmp / np.linalg.norm(tmp)
            except Exception:
                tmp = np.zeros(n_comps); tmp[0] = 1.0
                tmp = tmp - (tmp @ weights) * weights
                if np.linalg.norm(tmp) == 0:
                    u_res = np.zeros_like(weights)
                else:
                    u_res = tmp / np.linalg.norm(tmp)

            residual_bases_all[session] = u_res

            # Project test trials: LDA projection and residual projection (onto u_res)
            for t in sess_idx:
                if t in training_trials:
                    continue
                for f in range(n_frames):
                    frame_data = pca_projections[t, f, :n_comps]
                    projections[t, f] = np.dot(frame_data, weights)
                    # since u_res is orthogonal to weights, projecting frame_data onto u_res
                    # directly gives the residual 1D value
                    residual_projections[t, f] = np.dot(frame_data, u_res)

        all_projections[cv_idx] = projections
        all_residuals[cv_idx] = residual_projections

    # Safe average across CVs (ignore nan entries)
    # For numerical safety compute counts and sums to avoid warnings
    count_non_nan = np.sum(~np.isnan(all_projections), axis=0)
    sum_vals = np.nansum(all_projections, axis=0)
    avg_projections = np.full((n_trials, n_frames), np.nan)
    valid_mask = count_non_nan > 0
    if np.any(valid_mask):
        avg_projections[valid_mask] = sum_vals[valid_mask] / count_non_nan[valid_mask]

    count_non_nan_r = np.sum(~np.isnan(all_residuals), axis=0)
    sum_vals_r = np.nansum(all_residuals, axis=0)
    avg_residuals = np.full((n_trials, n_frames), np.nan)
    valid_mask_r = count_non_nan_r > 0
    if np.any(valid_mask_r):
        avg_residuals[valid_mask_r] = sum_vals_r[valid_mask_r] / count_non_nan_r[valid_mask_r]

    return avg_projections, avg_residuals, lda_weights_all, lda_models_all

# pertrubation direction projection keeping the timeseries for each trial
def signed_perturbation_projection(pca_projections, dev_on_frames, trial_sess_id, fp_frame_num=300):
    """
    Calculate signed compensation using full 300-frame windows.
    Uses random 50% of reaching trials for training.
    
    Parameters:
    -----------
    pca_projections: (n_trials, n_frames, n_comps) array
    dev_on_frames: (n_trials,) array of perturbation onset frames (NaN for controls)
    trial_sess_id: (n_trials,) array of session IDs
    fp_frame_num: Fixed window length (300 frames)
    """
    n_trials, n_frames, n_comps = pca_projections.shape
    perturb_mask = ~np.isnan(dev_on_frames)
    session_params = {}
    
    # 1. Compute session-specific perturbation vectors
    for session in np.unique(trial_sess_id):
        sess_mask = (trial_sess_id == session)
        reach_mask = np.arange(n_trials) < 140  # First 140 trials are reach trials
        
        # Get reaching trials
        reach_ctrl_idx = np.where(sess_mask & ~perturb_mask & reach_mask)[0]
        reach_pert_idx = np.where(sess_mask & perturb_mask & reach_mask)[0]
        
        # Split reaching trials for training
        ctrl_train_idx, _ = train_test_split(reach_ctrl_idx, test_size=0.5, random_state=42)
        pert_train_idx, _ = train_test_split(reach_pert_idx, test_size=0.5, random_state=42)
            
        # Extract and flatten control windows
        ctrl_windows = []
        for i in ctrl_train_idx:
            start = int(np.nanmean(dev_on_frames[perturb_mask]))
            start = max(0, min(start, n_frames - fp_frame_num))
            window = pca_projections[i, start:start+fp_frame_num, :].flatten()
            ctrl_windows.append(window)
        ctrl_mean = np.nanmean(ctrl_windows, axis=0)
        
        # Extract and flatten perturbed windows
        perturb_windows = []
        for i in pert_train_idx:
            start = int(dev_on_frames[i])
            start = max(0, min(start, n_frames - fp_frame_num))
            window = pca_projections[i, start:start+fp_frame_num, :].flatten()
            perturb_windows.append(window)
        perturb_mean = np.nanmean(perturb_windows, axis=0)
        
        # Compute normalized perturbation vector
        perturbation_vector = perturb_mean - ctrl_mean
        perturbation_vector /= np.linalg.norm(perturbation_vector)
        
        session_params[session] = {
            'ctrl_mean': ctrl_mean,
            'perturb_vector': perturbation_vector
        }
    
    # 2. Project all trials onto perturbation vectors
    signed_deviations = np.zeros(n_trials)
    
    for i in range(n_trials):
        session = trial_sess_id[i]
        params = session_params[session]
        
        # Get fixed window
        if perturb_mask[i]:
            start = int(dev_on_frames[i])
        else:
            start = int(np.nanmean(dev_on_frames[perturb_mask]))
            
        start = max(0, min(start, n_frames - fp_frame_num))
        window = pca_projections[i, start:start+fp_frame_num, :].flatten()
        
        # Center and project
        centered = window - params['ctrl_mean']
        signed_deviations[i] = np.dot(centered, params['perturb_vector'])
    
    return signed_deviations

def signed_perturbation_projection_mean(pca_projections, dev_on_frames, trial_sess_id, fp_frame_num=300):
    """
    Calculate signed compensation using mean activity in each window.
    Uses random 50% of reaching trials for training and sets their projections to NaN.
    
    Parameters:
    -----------
    pca_projections: (n_trials, n_frames, n_comps) array
    dev_on_frames: (n_trials,) array of perturbation onset frames (NaN for controls)
    trial_sess_id: (n_trials,) array of session IDs
    fp_frame_num: Fixed window length
    """
    n_trials, n_frames, n_components = pca_projections.shape
    
    # Select high variance components
    variance = np.var(pca_projections, axis=(0,1))
    var_ratio = variance / np.nansum(variance)
    high_var_comps = np.where(var_ratio >= 0.01)[0]  # Select components explaining >1% variance each
    n_comps = high_var_comps.size   
    print(f"Using {n_comps} components out of {n_components} based on variance threshold")
    
    perturb_mask = ~np.isnan(dev_on_frames)
    session_params = {}
    training_trials_all = []  # Keep track of all training trials
    
    # Initialize projections with NaNs
    signed_deviations = np.full(n_trials, np.nan)
    
    # 1. Compute session-specific perturbation vectors
    for session in np.unique(trial_sess_id):
        sess_mask = (trial_sess_id == session)
        reach_mask = np.arange(n_trials) < 140  # First 140 trials are reach trials
        
        # Get reaching trials
        reach_ctrl_idx = np.where(sess_mask & ~perturb_mask & reach_mask)[0]
        reach_pert_idx = np.where(sess_mask & perturb_mask & reach_mask)[0]
        
        # Split reaching trials for training
        ctrl_train_idx, _ = train_test_split(reach_ctrl_idx, test_size=0.5, random_state=42)
        pert_train_idx, _ = train_test_split(reach_pert_idx, test_size=0.5, random_state=42)
        
        # Store training trials
        training_trials_all.extend(ctrl_train_idx)
        training_trials_all.extend(pert_train_idx)
        
        # Extract and average control windows using high variance components
        ctrl_windows = []
        for i in ctrl_train_idx:
            start = int(np.nanmean(dev_on_frames[perturb_mask]))
            start = max(0, min(start, n_frames - fp_frame_num))
            window = pca_projections[i, start:start+fp_frame_num, high_var_comps]
            # Take mean across time
            ctrl_windows.append(np.nanmean(window, axis=0))
        ctrl_mean = np.nanmean(ctrl_windows, axis=0)
        
        # Extract and average perturbed windows using high variance components
        perturb_windows = []
        for i in pert_train_idx:
            start = int(dev_on_frames[i])
            start = max(0, min(start, n_frames - fp_frame_num))
            window = pca_projections[i, start:start+fp_frame_num, high_var_comps]
            # Take mean across time
            perturb_windows.append(np.nanmean(window, axis=0))
        perturb_mean = np.nanmean(perturb_windows, axis=0)
        
        # Compute normalized perturbation vector
        perturbation_vector = perturb_mean - ctrl_mean
        perturbation_vector /= np.linalg.norm(perturbation_vector)
        
        session_params[session] = {
            'ctrl_mean': ctrl_mean,
            'perturb_vector': perturbation_vector
        }
    
    # 2. Project only test trials and passive trials
    for i in range(n_trials):
        if i in training_trials_all:  # Skip training trials (keep as NaN)
            continue
            
        session = trial_sess_id[i]
        params = session_params[session]
        
        # Get fixed window and compute mean using high variance components
        if perturb_mask[i]:
            start = int(dev_on_frames[i])
        else:
            start = int(np.nanmean(dev_on_frames[perturb_mask]))
            
        start = max(0, min(start, n_frames - fp_frame_num))
        window = pca_projections[i, start:start+fp_frame_num, high_var_comps]
        # Take mean across time
        window_mean = np.nanmean(window, axis=0)
        
        # Center and project
        centered = window_mean - params['ctrl_mean']
        signed_deviations[i] = np.dot(centered, params['perturb_vector'])
    
    return signed_deviations

def path_length(pca_projections, dev_on_frames, n_comps=3, fp_frame_num = 300):
    n_trials, n_frames, n_comp = pca_projections.shape
    path_len = []
    
    # Determine perturbation trials where dev_on_frames is not NaN
    perturb_mask = ~np.isnan(dev_on_frames)
    perturb_trials_dev_on = dev_on_frames[perturb_mask]
    
    # Calculate average onset; default to 0 if no valid trials exist
    if perturb_trials_dev_on.size == 0:
        avg_onset = 0
    else:
        avg_onset = int(np.nanmean(perturb_trials_dev_on))
    
    for i in range(n_trials):
        trial_data = pca_projections[i]
        dev_on = dev_on_frames[i]
        
        # Determine the start frame for the perturbation period
        if np.isnan(dev_on):
            start_frame = avg_onset
        else:
            start_frame = int(dev_on)
        
        # Calculate end frame, ensuring it does not exceed the number of frames
        end_frame = start_frame + fp_frame_num  # 2 seconds * 240 Hz
        end_frame = min(end_frame, n_frames)
        
        # Extract the perturbation period segment
        segment = trial_data[start_frame:end_frame, :n_comps]
        
        # Compute path length for the segment
        if segment.shape[0] < 2:
            c_len = 0.0
        else:
            diffs = np.diff(segment, axis=0)
            norms = np.linalg.norm(diffs, axis=1)
            c_len = np.nansum(norms)
        
        path_len.append(c_len)
    
    return path_len

def oscillatory_energy_error(pca_projections, dev_on_frames, n_comps=3, sampling_rate=240, target_freq=10, freq_window=1):
    """
    Calculate 10Hz oscillation energy during the perturbation period for 3D input (n_trials, n_frames, n_features).
    """
    n_trials, n_frames, n_comp = pca_projections.shape
    energies = np.zeros(n_trials)
    
    # Determine valid perturbation trials and compute average onset
    perturb_mask = ~np.isnan(dev_on_frames)
    perturb_trials_dev_on = dev_on_frames[perturb_mask]
    
    if len(perturb_trials_dev_on) == 0:
        avg_onset = 0  # Default if no valid trials
    else:
        avg_onset = int(np.nanmean(perturb_trials_dev_on))
    
    for trial_idx in range(n_trials):
        dev_on = dev_on_frames[trial_idx]
        
        # Determine perturbation window start frame
        if np.isnan(dev_on):
            start_frame = avg_onset
        else:
            start_frame = int(dev_on)
        
        # Calculate end frame (2-second window)
        end_frame = start_frame + int(2 * sampling_rate)
        end_frame = min(end_frame, n_frames)
        segment_length = end_frame - start_frame
        
        # Skip energy calculation if the segment is invalid
        if segment_length <= 0:
            energies[trial_idx] = 0.0
            continue
        
        # Frequency analysis for the segment
        frequencies = np.fft.fftfreq(segment_length, d=1/sampling_rate)
        freq_mask = (np.abs(frequencies - target_freq) <= freq_window)
        
        # Skip if no relevant frequencies in the segment
        if not np.any(freq_mask):
            energies[trial_idx] = 0.0
            continue
        
        # Compute energy for each component in the segment
        trial_energy = 0.0
        for comp_idx in range(n_comps):
            ts = pca_projections[trial_idx, start_frame:end_frame, comp_idx]
            ts_detrended = signal.detrend(ts)
            fft_vals = np.fft.fft(ts_detrended)
            psd = np.abs(fft_vals) ** 2
            trial_energy += np.nansum(psd[freq_mask])
        
        # Average energy across components
        energies[trial_idx] = trial_energy / n_comps
    
    return energies


# LDA with full temporal trajectories
def signed_lda_temporal(pca_projections, dev_on_frames, trial_sess_id, 
                       fp_frame_num=300):
    """
    Session-specific LDA using full temporal trajectories.
    Uses only 50% of reaching trials for training.
    
    Parameters:
    -----------
    pca_projections : array-like (n_trials x n_frames x n_components)
        PCA projected neural data
    dev_on_frames : array-like
        Perturbation onset frames (NaN for controls)
    trial_sess_id : array-like
        Session ID for each trial
    fp_frame_num : int
        Number of frames to use after perturbation onset
    """

    
    n_trials, n_frames, n_comps = pca_projections.shape
    y = ~np.isnan(dev_on_frames)
    projections = np.zeros(n_trials)
    virtual_dev_on = int(np.nanmean(dev_on_frames[y]))

    for session in np.unique(trial_sess_id):
        # Get session indices and trials for reaching only
        sess_mask = (trial_sess_id == session)
        reach_mask = np.arange(n_trials) < 140  # First 140 trials are reach trials
        sess_reach_mask = sess_mask & reach_mask
        sess_idx = np.where(sess_reach_mask)[0]
        
        # Extract temporal features for reaching trials
        X_sess = []
        y_sess = []
        for i in sess_idx:
            start = int(dev_on_frames[i]) if y[i] else virtual_dev_on
            start = max(0, min(start, n_frames - fp_frame_num))
            window = pca_projections[i, start:start+fp_frame_num, :]
            X_sess.append(window.reshape(-1))  # Flatten to (fp_frame_num*n_comps,)
            y_sess.append(y[i])
        X_sess = np.array(X_sess)
        y_sess = np.array(y_sess)
        
        # Split reaching trials into train/test
        train_idx, _ = train_test_split(np.arange(len(sess_idx)), 
                                      test_size=0.5, 
                                      stratify=y_sess,
                                      random_state=42)
        
        # Train LDA on temporal patterns using only training trials
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_sess[train_idx], y_sess[train_idx])
        
        # Normalize LDA weights
        weights = lda.coef_.T
        weights = weights / np.linalg.norm(weights)
        
        # Project ALL trials in session (including non-reach trials)
        sess_all_idx = np.where(sess_mask)[0]
        X_all = []
        for i in sess_all_idx:
            start = int(dev_on_frames[i]) if y[i] else virtual_dev_on
            start = max(0, min(start, n_frames - fp_frame_num))
            window = pca_projections[i, start:start+fp_frame_num, :]
            X_all.append(window.reshape(-1))
        X_all = np.array(X_all)
        
        # Project using normalized weights
        projections[sess_all_idx] = np.dot(X_all, weights).squeeze()

    return projections


def signed_lda_pks(pca_projections, dev_on_frames, trial_sess_id, fp_frame_num=300):
    """
    Session-specific LDA using peak activity in the window.
    Projects whole trial first, then finds peaks in the projection.
    Uses 50% of trials for training and sets their projections to NaN.
    """
    
    n_trials, n_frames, n_components = pca_projections.shape
    # Select components based on variance
    variance = np.var(pca_projections, axis=(0,1))
    var_ratio = variance / np.nansum(variance)
    n_comps = np.nansum(var_ratio >= 0.01)  # Count components explaining >1% variance each
    print(f"Using first {n_comps} components out of {n_components} based on variance threshold")

    y = ~np.isnan(dev_on_frames)
    # Initialize projections with NaNs
    projections = np.full(n_trials, np.nan)
    virtual_dev_on = int(np.nanmean(dev_on_frames[y]))

    for session in np.unique(trial_sess_id):
        # Get session indices and trials for reaching only
        sess_mask = (trial_sess_id == session)
        reach_mask = np.arange(n_trials) < 140  # First 140 trials are reach trials
        sess_reach_mask = sess_mask & reach_mask
        sess_idx = np.where(sess_reach_mask)[0]
        
        # Extract peak activity in window for reaching trials
        X_sess = []
        y_sess = []
        valid_trials = []  # Keep track of valid trials
        for i in sess_idx:
            start = int(dev_on_frames[i]) if y[i] else virtual_dev_on
            start = max(0, min(start, n_frames - fp_frame_num))
            # Get window for first n_comps components
            window = pca_projections[i, start:start+fp_frame_num, :n_comps]
            # Find peaks across time for each component
            peak_idx = np.argmax(np.abs(window), axis=0)
            # Get peak values
            peak_values = window[peak_idx, np.arange(n_comps)]
            X_sess.append(peak_values)
            y_sess.append(y[i])
            valid_trials.append(i)
        
        X_sess = np.array(X_sess)
        y_sess = np.array(y_sess)
        valid_trials = np.array(valid_trials)
        
        # Split reaching trials into train/test
        train_idx, test_idx = train_test_split(np.arange(len(sess_idx)), 
                                             test_size=0.5, 
                                             stratify=y_sess,
                                             random_state=42)
        
        # Get actual trial indices for training and testing
        training_trials = valid_trials[train_idx]
        test_trials = valid_trials[test_idx]
        
        # Train LDA on peak activity patterns using only training trials
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_sess[train_idx], y_sess[train_idx])
        
        # Normalize LDA weights
        weights = lda.coef_[0]  # Get first row for binary classification
        weights = weights / np.linalg.norm(weights)
        
        # Project only test trials and passive trials
        sess_test_idx = np.where(sess_mask)[0]
        for i in sess_test_idx:
            if i in training_trials:  # Skip training trials (keep as NaN)
                continue
                
            # Project entire trial
            trial_data = pca_projections[i, :, :n_comps]  # Get whole trial
            trial_projection = np.dot(trial_data, weights)  # Project whole trial
            
            # Find peak in the window of interest
            start = int(dev_on_frames[i]) if y[i] else virtual_dev_on
            start = max(0, min(start, n_frames - fp_frame_num))
            end = start + fp_frame_num
            window_projection = trial_projection[start:end]
            
            # Get peak value from projected window
            peak_idx = np.argmax(np.abs(window_projection))
            projections[i] = window_projection[peak_idx]

    return projections



# -----------------------------------------------------------------------------------------------------------
# Kinematic Subspaces Helper functions
# ------------------------------------------------------------------------------------------------------------
def preprocess_signal(signal_input, frame_slice, trial_blocks):
    """
    Preprocess neural signal matrix by averaging trials within blocks,
    concatenating blocks, and normalizing per neuron.

    Parameters
    ----------
    signal_input : np.ndarray
        Neural data, shape (n_neuron, n_trial, n_frame).
    frame_slice : slice
        Frame indices to select (e.g., FRAME_SLICE).

    Returns
    -------
    N_norm : np.ndarray
        Preprocessed neural matrix, shape (n_neuron, total_frames).
        - Trials are averaged within each condition block (Pre, Dev, Post).
        - Concatenated across blocks in time.
        - Each neuron is range-normalized and mean-subtracted.

    N_norm_3d : np.ndarray
        Normalized per-trial data, shape (n_neuron, n_trial, n_selected_frames).
        - Each neuron normalized across all trials and frames first.
    """
    # --- Slice frames
    signal_input = signal_input[:, :, frame_slice]  # (n_neuron, n_trial, n_selected_frames)

    # --- Normalize first (per neuron across all trials and frames)
    neuron_min = signal_input.min(axis=(1, 2), keepdims=True)
    neuron_max = signal_input.max(axis=(1, 2), keepdims=True)
    neuron_range = neuron_max - neuron_min
    neuron_range[neuron_range < 1e-6] = 1.0  # prevent div by zero

    N_norm_3d = (signal_input - signal_input.mean(axis=(1, 2), keepdims=True)) / neuron_range

    # --- Average across trials within each block
    N_blocks = []
    for cond, trial_idx in trial_blocks.items():
        block_data = N_norm_3d[:, trial_idx, :]  # (n_neuron, n_trials, n_frames)
        block_mean = np.mean(block_data, axis=1) # (n_neuron, n_frames)
        N_blocks.append(block_mean)

    # --- Concatenate across blocks
    N_norm = np.concatenate(N_blocks, axis=1)  # (n_neuron, total_frames)

    return N_norm, N_norm_3d

def compute_potent_null_with_lag(N_pca, M_pca, ctrl_frame_idx=None, dev_frame_idx=None, lag_range=15):
    """
    Compute potent and null subspaces using PCA-projected neural/behavior data.
    N_pca and M_pca should already be projected into their respective PCs.

    Returns potent/null bases, R² for Ctrl fit, R² for Dev (shifted by best lag),
    and R² for perturbation effect (Dev-Ctrl).

    Parameters
    ----------
    N_pca : np.ndarray
        Neural PCs (n_neural_pcs × total_frames)
    M_pca : np.ndarray
        Behavior PCs (n_behavior_pcs × total_frames)
    ctrl_frame_idx : array-like, optional
        Indices of frames to use for fitting potent/null
    dev_frame_idx : array-like, optional
        Indices of frames to compute R² on Dev/perturbation frames
    lag_range : int
        Max ± lag for alignment

    Returns
    -------
    dict
        Potent/null bases, R² for Ctrl and Dev (shifted by best lag), W, trajectories
    """
    best_r2, best_lag = -np.inf, 0
    best_potent, best_null, best_W = None, None, None

    # --- Select only control frames for fitting
    if ctrl_frame_idx is not None:
        N_fit = N_pca[:, ctrl_frame_idx]
        M_fit = M_pca[:, ctrl_frame_idx]
    else:
        N_fit = N_pca
        M_fit = M_pca

    # --- Lag search on fitting frames
    for lag in range(-lag_range, lag_range + 1):
        if lag < 0:
            N_shift = N_fit[:, -lag:]
            M_shift = M_fit[:, :M_fit.shape[1] + lag]
        elif lag > 0:
            N_shift = N_fit[:, :-lag]
            M_shift = M_fit[:, lag:]
        else:
            N_shift, M_shift = N_fit, M_fit

        if N_shift.shape[1] <= 5:
            continue

        ridge = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=None, fit_intercept=False)
        ridge.fit(N_shift.T, M_shift.T)
        W_tilde = ridge.coef_

        M_pred = ridge.predict(N_shift.T).T
        r2 = r2_score(M_shift.T, M_pred.T, multioutput='variance_weighted')

        if r2 > best_r2:
            best_r2 = r2
            best_lag = lag
            best_W = W_tilde

            # SVD to get potent/null
            U, S, Vt = svd(W_tilde)
            potent_basis = Vt[:M_shift.shape[0], :].T
            null_basis   = Vt[M_shift.shape[0]:, :].T
            best_potent, best_null = potent_basis, null_basis

    # --- Project full data onto potent/null
    traj_potent = best_potent.T @ N_pca
    traj_null   = best_null.T   @ N_pca

    # --- Compute R² on Dev frames and perturbation effect
    r2_dev = None
    r2_perturb = None
    if dev_frame_idx is not None and ctrl_frame_idx is not None:
        # Ctrl frames
        N_ctrl = N_pca[:, ctrl_frame_idx]
        M_ctrl = M_pca[:, ctrl_frame_idx]

        # Dev frames
        N_dev  = N_pca[:, dev_frame_idx]
        M_dev  = M_pca[:, dev_frame_idx]

        # Apply best lag
        if best_lag < 0:
            N_dev_shift  = N_dev[:, -best_lag:]
            M_dev_shift  = M_dev[:, :M_dev.shape[1] + best_lag]
            N_ctrl_shift = N_ctrl[:, -best_lag:]
            M_ctrl_shift = M_ctrl[:, :M_ctrl.shape[1] + best_lag]
        elif best_lag > 0:
            N_dev_shift  = N_dev[:, :-best_lag]
            M_dev_shift  = M_dev[:, best_lag:]
            N_ctrl_shift = N_ctrl[:, :-best_lag]
            M_ctrl_shift = M_ctrl[:, best_lag:]
        else:
            N_dev_shift, M_dev_shift = N_dev, M_dev
            N_ctrl_shift, M_ctrl_shift = N_ctrl, M_ctrl

        # Dev R² with best lag
        M_pred_dev = best_W @ N_dev_shift
        r2_dev = r2_score(M_dev_shift.T, M_pred_dev.T, multioutput='variance_weighted')

        # Perturbation R²: ΔM vs W ΔN
        ΔM = M_dev_shift - M_ctrl_shift
        ΔN = N_dev_shift - N_ctrl_shift
        M_pred_Δ = best_W @ ΔN
        r2_perturb = r2_score(ΔM.T, M_pred_Δ.T, multioutput='variance_weighted')

    return {
        "potent_basis": best_potent,
        "null_basis": best_null,
        "best_lag": best_lag,
        "best_r2": best_r2,
        "r2_dev": r2_dev,
        "r2_perturb": r2_perturb,
        "W_tilde": best_W,
        "traj_potent": traj_potent,
        "traj_null": traj_null
    }


def project_to_potent_basis(signal_input, potent_basis):
    """
    Project neural data onto the potent basis.

    Args:
        signal_input (np.ndarray): Neural data of shape (n_neurons, n_trials, n_frames).
        potent_basis (np.ndarray): Potent basis of shape (n_neurons, rank).

    Returns:
        np.ndarray: Projected data of shape (rank, n_trials, n_frames).
    """
    # Reshape signal_input to (n_neurons, n_trials * n_frames) for matrix multiplication
    n_neurons, n_trials, n_frames = signal_input.shape
    reshaped_signal = signal_input.reshape(n_neurons, -1)  # Shape: (n_neurons, n_trials * n_frames)

    # Project onto the potent basis
    projected_signal = potent_basis.T @ reshaped_signal  # Shape: (rank, n_trials * n_frames)

    # Reshape back to (rank, n_trials, n_frames)
    projected_signal = projected_signal.reshape(potent_basis.shape[1], n_trials, n_frames)

    return projected_signal
