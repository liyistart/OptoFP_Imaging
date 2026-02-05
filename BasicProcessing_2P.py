import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

def get_spiky_mask(data, threshold=3, min_spikes=5):
    """
    Identify neurons with spiky activity based on threshold crossings in both positive and negative directions.

    Parameters:
    -----------
    data : np.ndarray
        2D array (n_neurons, n_timepoints) of activity (e.g., block_results['blank'])
    threshold : float
        Value above which a peak is considered a spike (applies to both positive and negative peaks)
    min_spikes : int
        Minimum number of spikes to be considered "spiky"

    Returns:
    --------
    spiky_mask : np.ndarray
        Boolean mask of shape (n_neurons,) where True indicates spiky neurons
    """
    n_neurons = data.shape[0]
    spiky_mask = np.zeros(n_neurons, dtype=bool)
    for i in range(n_neurons):
        # Find positive peaks above threshold
        pos_peaks, _ = find_peaks(data[i], height=threshold)
        # Find negative peaks below -threshold
        neg_peaks, _ = find_peaks(-data[i], height=threshold)
        # Combine both directions
        total_spikes = len(pos_peaks) + len(neg_peaks)
        if total_spikes >= min_spikes:
            spiky_mask[i] = True
    return spiky_mask

def smooth_2d_data(data, window=10):
    """
    Smooth 2D ROC data along the time dimension
    
    Parameters:
    -----------
    data : numpy.ndarray
        Shape (n_neurons, time_points)
    window : int
        Window size for smoothing
        
    Returns:
    --------
    smoothed_data : numpy.ndarray 
        Smoothed data with same shape as input
    """
    n_neurons = data.shape[0]
    smoothed_data = np.zeros_like(data)
    sigma = window / 5.0  # Convert window to sigma for gaussian
    
    for n in range(n_neurons):
        smoothed_data[n, :] = gaussian_filter1d(data[n, :], sigma, mode='reflect')
    
    return smoothed_data

def smooth_3d_data(data, window=10, method='gaussian'):
    """
    Smooth 3D data array along the time dimension with improved edge handling.
    Parameters:
    -----------
    data : numpy.ndarray
        Shape (n_neurons, n_trials, trial_len)
    window : int
        Window size for smoothing (default: 5)
    method : str
        'gaussian' or 'moving' for different smoothing methods
    Returns:
    --------
    smoothed_data : numpy.ndarray
        Smoothed data with same shape as input
    """
    n_neurons, n_trials, trial_len = data.shape
    smoothed_data = np.zeros_like(data)
    if method == 'gaussian':
        sigma = window / 5.0
        for n in range(n_neurons):
            for t in range(n_trials):
                smoothed_data[n, t, :] = gaussian_filter1d(
                    data[n, t, :], sigma, mode='reflect'
                )
    elif method == 'moving':
        for n in range(n_neurons):
            for t in range(n_trials):
                smoothed_data[n, t, :] = savgol_filter(
                    data[n, t, :],
                    window_length=window,
                    polyorder=2,
                    mode='interp'
                )
    return smoothed_data

# handling NaNs in 3D matrix
def replace_nans_in_rows_3d(matrix_3d):
    """
    Replace NaNs in a 3D matrix by interpolating each row in the 2D slices independently.
    Parameters:
    matrix_3d (numpy.ndarray): 3D input matrix with shape (depth, rows, cols).
    Returns:
    numpy.ndarray: 3D matrix with NaNs replaced.
    """
    filled_matrix = np.zeros_like(matrix_3d)
    depth, rows, cols = matrix_3d.shape
    for d in range(depth):
        for r in range(rows):
            # Get the row, handle NaNs with interpolation
            row = pd.Series(matrix_3d[d, r, :])
            row_filled = row.interpolate(method='nearest', limit_direction='both').fillna(0)
            filled_matrix[d, r, :] = row_filled.values

    return filled_matrix

# Compute z-scores using per-trial baseline
def compute_trial_zscore(data, pre_frames):
    """
    Z-score data using per-trial baseline period
    
    Parameters:
    -----------
    data : numpy.ndarray
        Shape (n_neurons, n_trials, trial_len)
    pre_frames : int
        Number of frames to use for baseline
        
    Returns:
    --------
    tuple:
        zscored_data, m_base, std_base
    """
    # Get baseline mean and std for each trial
    m_base = np.nanmean(data[:, :, 0:pre_frames], axis=2)  # shape: (n_neurons, n_trials)
    std_base = np.nanstd(data[:, :, 0:pre_frames], axis=2) # shape: (n_neurons, n_trials)
    
    # Add singleton dimension for broadcasting across time
    m_base_exp = m_base[:, :, None]   # shape: (n_neurons, n_trials, 1)
    std_base_exp = std_base[:, :, None] # shape: (n_neurons, n_trials, 1)
    
    # Z-score each trial using its own baseline
    zscored_data = (data - m_base_exp) / std_base_exp
    
    return zscored_data, m_base, std_base

def compute_trial_dfbyf(data, pre_frames, threshold=5):
    """
    Compute df/f data using per-trial baseline period with zero handling
    
    Parameters:
    -----------
    data : numpy.ndarray
        Shape (n_neurons, n_trials, trial_len)
    pre_frames : int
        Number of frames to use for baseline
    threshold : float
        Minimum value for baseline mean
        
    Returns:
    --------
    tuple:
        dfbyf_data, m_base, std_base, zero_mask
    """
    # Get baseline mean and std for each trial
    m_all = np.nanmax(data[:, :, :], axis=(2))  # shape: (n_neurons, n_trials)
    m_base = np.nanmean(data[:, :, 0:pre_frames], axis=(2))  # shape: (n_neurons, n_trials)
    std_base = np.nanstd(data[:, :, 0:pre_frames], axis=(2))  # shape: (n_neurons, n_trials)
    
    # Create mask for near-zero baseline values
    zero_mask = m_base < threshold  # shape: (n_neurons, n_trials)
    
    # Handle near-zero baseline values directly in 2D
    m_base_safe = m_base.copy()
    # Replace with maximum of m_all or threshold where baseline is too low
    m_base_safe[zero_mask] = np.nan#m_all[zero_mask]
    
    # Expand dimensions for broadcasting with data
    m_base_safe_exp = m_base_safe[:, :, np.newaxis]
    
    # Compute df/f using safe baseline values
    dfbyf_data = (data - m_base_safe_exp) / m_base_safe_exp
    
    return dfbyf_data, m_base, std_base, zero_mask

# concatanate all sessions
# def concatenate_sessions(session_list, axis=0, select_trials=140):
#     session_array = []
#     session_ids = []
    
#     # Iterate over sessions while tracking the index for session IDs
#     for session_idx, sess_key in enumerate(session_list):
#         # Get the session data (NumPy array)
#         session_data = session_list[sess_key]
#         # Select the first 'select_trials' trials from the session
#         if np.size(select_trials) == 0:
#             selected_trials = session_data
#         else:
#             selected_trials = session_data[:select_trials]
#         session_array.append(selected_trials)
        
#         # Generate session IDs for each trial in this session
#         num_trials = selected_trials.shape[0]  # Now safe (session_data is a NumPy array)
#         session_ids.append(np.full(num_trials, session_idx))
    
#     # Concatenate all trials and their session IDs
#     concatenated_array = np.concatenate(session_array, axis=axis)
#     concatenated_ids = np.concatenate(session_ids, axis=0)
    
#     return concatenated_array, concatenated_ids

def concatenate_sessions(session_list, axis=0, select_trials=None):
    """
    Concatenate session data arrays with flexible trial selection
    
    Parameters:
    -----------
    session_list : dict
        Dictionary of session data arrays
    axis : int
        Axis along which to concatenate (default=0)
    select_trials : int or None
        Number of trials to select from each session
        If None, use all trials from each session
        
    Returns:
    --------
    tuple: (concatenated_array, concatenated_ids)
        concatenated_array: Combined data from all sessions
        concatenated_ids: Session ID for each row/trial
    """
    session_array = []
    session_ids = []
    
    # Input validation
    if not isinstance(session_list, dict):
        raise TypeError("session_list must be a dictionary")
    
    # Iterate over sessions
    for session_idx, sess_key in enumerate(session_list):
        # Get session data
        session_data = session_list[sess_key]
        
        if not isinstance(session_data, np.ndarray):
            raise TypeError(f"Session {sess_key} data must be numpy array")
        
        # Select trials based on input parameter
        if select_trials is None:
            # Use all trials
            selected_trials = session_data
        else:
            # Validate select_trials
            if not isinstance(select_trials, (int, np.integer)):
                raise TypeError("select_trials must be an integer or None")
            if select_trials <= 0:
                raise ValueError("select_trials must be positive")
                
            # Select subset of trials
            selected_trials = session_data[:min(select_trials, session_data.shape[0])]
        
        # Store selected trials and generate session IDs
        session_array.append(selected_trials)
        session_ids.append(np.full(selected_trials.shape[0], session_idx))
        
        # Print diagnostic information
        print(f"Session {sess_key}: {selected_trials.shape}")
    
    # Concatenate data
    try:
        concatenated_array = np.concatenate(session_array, axis=axis)
        concatenated_ids = np.concatenate(session_ids, axis=0)
        
        print(f"\nConcatenation summary:")
        print(f"Total sessions: {len(session_list)}")
        print(f"Total samples: {len(concatenated_ids)}")
        print(f"Output shape: {concatenated_array.shape}")
        
        return concatenated_array, concatenated_ids
        
    except Exception as e:
        print(f"Concatenation failed: {str(e)}")
        raise

def concatenate_dfbyf_sessions(results):
    """
    Concatenate dfbyf data across sessions while maintaining trial structure
    
    Parameters:
    -----------
    results : dict
        Dictionary of results from each session
        'f_cells': f_cells_align,
        'dfbyf': signal_input, # returning the smoothed data
        'spks_align': spks_align, # shaped (n_cells, n_trials, n_frames)
        'events': first_on_events,
        'auroc': auroc_results,
        'blocks': block_results,
        'dev_on_frames_2p': dev_on_frames_2p,
        'keep_mask': iscell,
        'corrected_fs_2p': first_on_events['fs_2p'],
        
    Returns:
    --------
    dict:
        'dfbyf': concatenated dfbyf data (n_total_neurons, n_trials, n_timepoints)
        'spks': concatenated spks data (n_total_neurons, n_trials, n_timepoints)
        'corrected_fs_2p': corrected fs_2p data (n_trials,)
        'dev_on_frames_2p': concatenated dev_on_frames
        'neuron_session_map': array indicating which session each neuron came from
    """
    # Get dimensions
    n_trials = results[list(results.keys())[0]]['dfbyf'].shape[1]
    n_timepoints = results[list(results.keys())[0]]['dfbyf'].shape[2]
    
    # Count total neurons
    total_neurons = sum(session_data['dfbyf'].shape[0] for session_data in results.values())
    
    # Initialize arrays
    all_dfbyf = np.zeros((total_neurons, n_trials, n_timepoints))
    all_spks = np.zeros((total_neurons, n_trials, n_timepoints))  # Assuming spks is also needed
    all_dev_on = []
    all_fs_2p = [] # Assuming fs_2p is also needed
    neuron_session_map = []
    
    # Concatenate data
    current_neuron = 0
    for session, data in results.items():
        n_neurons = data['dfbyf'].shape[0]
        
        # Copy dfbyf data
        all_dfbyf[current_neuron:current_neuron + n_neurons] = data['dfbyf']
        
        # Copy spks data if available
        all_spks[current_neuron:current_neuron + n_neurons] = data['spks_align']

        all_dev_on.append(data['dev_on_frames_2p'])
        all_fs_2p.append(data['corrected_fs_2p'])
            
        # Track which neurons belong to which session
        neuron_session_map.extend([session] * n_neurons)
        
        current_neuron += n_neurons

    # Concatenate lists into 1D arrays
    all_dev_on = np.concatenate(all_dev_on)
    all_fs_2p = np.concatenate(all_fs_2p)
    
    return {
        'dfbyf': all_dfbyf,
        'spks': all_spks,
        'dev_on_frames_2p': all_dev_on,
        'corrected_fs_2p': all_fs_2p,
        'neuron_session_map': np.array(neuron_session_map)
    }

def concatenate_dict_sessions(session_dict, select_fields=None):
    """
    Concatenate multiple session results along dimension 0 for each field
    
    Parameters:
    -----------
    session_dict : dict
        Dictionary of dictionaries, where each inner dict contains results for one session
    select_fields : list or None
        List of fields to concatenate. If None, concatenate all fields
        
    Returns:
    --------
    dict:
        Concatenated results dictionary
        session_ids: array indicating which session each row came from
    """
    # Initialize output dictionary and lists
    concatenated_results = {}
    session_ids = []
    
    # Get all fields if none specified
    if select_fields is None:
        first_session = list(session_dict.values())[0]
        select_fields = list(first_session.keys())
    
    # Initialize dictionary for each field
    for field in select_fields:
        concatenated_results[field] = []
    
    # Iterate through sessions
    for session_idx, sess_key in enumerate(session_dict):
        session_data = session_dict[sess_key]
        
        # Get number of neurons from first non-None field
        n_neurons = None
        for field in select_fields:
            if field in session_data and session_data[field] is not None:
                # Convert to numpy array if it's a list
                if isinstance(session_data[field], list):
                    session_data[field] = np.array(session_data[field])
                n_neurons = session_data[field].shape[0]
                break
                
        if n_neurons is not None:
            session_ids.append(np.full(n_neurons, session_idx))
            
            # Concatenate each field
            for field in select_fields:
                if field in session_data and session_data[field] is not None:
                    # Convert to numpy array if it's a list
                    if isinstance(session_data[field], list):
                        session_data[field] = np.array(session_data[field])
                    concatenated_results[field].append(session_data[field])
    
    # Concatenate along dimension 0 for each field
    for field in select_fields:
        if concatenated_results[field]:  # if list is not empty
            concatenated_results[field] = np.concatenate(concatenated_results[field], axis=0)
        else:
            concatenated_results[field] = None
    
    # Concatenate session IDs
    concatenated_results['session_ids'] = np.concatenate(session_ids, axis=0)
    
    return concatenated_results