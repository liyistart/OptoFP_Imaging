import numpy as np
import matplotlib.pyplot as plt
import sys
# sys.path.insert(0,'/home/sp645/external_packages/dynamax/dynamax')
# from dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression


'''
===================
Ridge helper
===================
'''

def _stack_lags(arr, max_lag, start, end):
    """
    Return [arr_t, arr_{t-1}, ..., arr_{t-max_lag}] with shape (N, (max_lag+1)*D).
    """
    cols = [arr[start - k:end - k] for k in range(max_lag + 1)]
    return np.concatenate(cols, axis=1)

def _ridge_solve(Phi, Y, lam, n_unpenalized=1):
    F = Phi.shape[1]
    # diagonal penalty, zero out the intercept columns
    pen = np.ones((F,))
    if n_unpenalized > 0:
        pen[-n_unpenalized:] = 0.0
    D = np.diag(pen)

    G = Phi.T @ Phi + lam * D          # (F,F)
    RHS = Phi.T @ Y                    # (F,output_dim)
    # Solve G Theta = RHS
    Theta = np.linalg.solve(G, RHS)
    return Theta

'''
===================
Help function for building lag block in regression
===================
'''

def _build_lag_design_one_trial(z, x, u, L, J, add_intercept=True):
    """
    z: (T, z_dim)
    x: (T, x_dim)
    u: (T,) or (T,1)
    Returns:
        Phi_z, Z_next, Phi_x, X_next
    """
    T, z_dim = z.shape
    _, x_dim = x.shape
    u = u.reshape(-1, 1) if u.ndim == 1 else u  # (T,1)

    maxlag = max(L, J)
    t0, t1 = maxlag, T - 1     # predict at t in [t0 .. t1-1], target is t+1
    N = t1 - t0

    Z_t    = z[t0:t1]          # (N, z_dim)
    Z_next = z[t0+1:t1+1]      # (N, z_dim)
    X_t    = x[t0:t1]          # (N, x_dim)
    X_next = x[t0+1:t1+1]      # (N, x_dim)

    X_lags = _stack_lags(x, L, start=t0, end=t1)      # (N, (L+1)*x_dim)
    Z_lags = _stack_lags(z, J, start=t0, end=t1)      # (N, (J+1)*z_dim)
    U_lags = _stack_lags(u, J, start=t0, end=t1)      # (N, (J+1)*1)

    # u-gated z-lags: [u_t*z_t, u_{t-1}*z_{t-1}, ..., u_{t-J}*z_{t-J}]
    UZ_blocks = []
    for k in range(J + 1):
        uk = U_lags[:, [k]]                               # (N,1)
        zk = Z_lags[:, k*z_dim:(k+1)*z_dim]               # (N,z_dim)
        UZ_blocks.append(uk * zk)
    UZ_lags = np.concatenate(UZ_blocks, axis=1)          # (N, (J+1)*z_dim)

    # -------- z-equation --------
    Phi_z_parts = [Z_t, X_lags]
    if add_intercept:
        Phi_z_parts.append(np.ones((N, 1)))
    Phi_z = np.concatenate(Phi_z_parts, axis=1)          # (N, ...)

    # -------- x-equation --------
    Phi_x_parts = [X_t, Z_lags, UZ_lags]
    if add_intercept:
        Phi_x_parts.append(np.ones((N, 1)))
    Phi_x = np.concatenate(Phi_x_parts, axis=1)          # (N, ...)

    return Phi_z, Z_next, Phi_x, X_next, Z_t


'''
===================
Help function for fitting and evaluating ridge regression full model
===================
'''

def fit_lag_model_ridge_multitrial(
    z_list, x_list, u_list,  
    L, J,
    y_list=None,    
    lam_z=1e-2,
    lam_x=1e-2,
    lam_y=1e-2,
    add_intercept=True
):
    """
    Trial_splitted Ridge.
    """

    # ---------- 1. Build and stack designs over trials ----------
    Phi_z_all = []
    Z_next_all = []
    Phi_x_all = []
    X_next_all = []
    Z_t_all = []
    Y_t_all = [] if y_list is not None else None

    for i, (z, x, u) in enumerate(zip(z_list, x_list, u_list)):
        Phi_z_i, Z_next_i, Phi_x_i, X_next_i, Z_t_i = \
            _build_lag_design_one_trial(z, x, u, L, J, add_intercept=add_intercept)

        Phi_z_all.append(Phi_z_i)
        Z_next_all.append(Z_next_i)
        Phi_x_all.append(Phi_x_i)
        X_next_all.append(X_next_i)
        Z_t_all.append(Z_t_i)

        if y_list is not None:
            y = y_list[i]
            maxlag = max(L, J)
            t0, t1 = maxlag, y.shape[0] - 1
            Y_t_all.append(y[t0:t1])

    Phi_z = np.vstack(Phi_z_all)      # (N_total, ...)
    Z_next = np.vstack(Z_next_all)    # (N_total, z_dim)
    Phi_x = np.vstack(Phi_x_all)      # (N_total, ...)
    X_next = np.vstack(X_next_all)    # (N_total, x_dim)
    Z_t_cat = np.vstack(Z_t_all)      # (N_total, z_dim)
    if y_list is not None:
        Y_t_cat = np.vstack(Y_t_all)  # (N_total, y_dim)

    N_total, z_dim = Z_next.shape
    _, x_dim = X_next.shape

    # ---------- 2. z-ridge ----------
    Theta_z = _ridge_solve(
        Phi_z, Z_next,
        lam=lam_z,
        n_unpenalized=1 if add_intercept else 0
    )
    idx = 0
    A = Theta_z[idx:idx+z_dim, :].T
    idx += z_dim
    B_flat_T = Theta_z[idx:idx+(L+1)*x_dim, :].T
    idx += (L+1)*x_dim
    d1 = Theta_z[idx, :] if add_intercept else np.zeros((z_dim,))

    B_lag = B_flat_T.reshape(z_dim, L+1, x_dim).transpose(1, 0, 2)  # (L+1, z_dim, x_dim)

    # ---------- 3. x-ridge ----------
    Theta_x = _ridge_solve(
        Phi_x, X_next,
        lam=lam_x,
        n_unpenalized=1 if add_intercept else 0
    )
    idx = 0
    P = Theta_x[idx:idx+x_dim, :].T
    idx += x_dim

    K0_flat_T = Theta_x[idx:idx+(J+1)*z_dim, :].T
    idx += (J+1)*z_dim
    K1_flat_T = Theta_x[idx:idx+(J+1)*z_dim, :].T
    idx += (J+1)*z_dim

    d2 = Theta_x[idx, :] if add_intercept else np.zeros((x_dim,))

    K0_lag = K0_flat_T.reshape(x_dim, J+1, z_dim).transpose(1, 0, 2)
    K1_lag = K1_flat_T.reshape(x_dim, J+1, z_dim).transpose(1, 0, 2)

    results = dict(
        A=A, P=P,
        B_lag=B_lag,
        K0_lag=K0_lag,
        K1_lag=K1_lag,
        d_z=d1,
        d_x=d2,
    )

    # ---------- 4. optional readout (y) ----------
    if y_list is not None:
        Phi_y_parts = [Z_t_cat]
        if add_intercept:
            Phi_y_parts.append(np.ones((N_total, 1)))
        Phi_y = np.concatenate(Phi_y_parts, axis=1)

        Theta_y = _ridge_solve(
            Phi_y, Y_t_cat,
            lam=lam_y,
            n_unpenalized=1 if add_intercept else 0
        )
        if add_intercept:
            C = Theta_y[:-1, :].T
            b = Theta_y[-1, :]
        else:
            C = Theta_y.T
            b = np.zeros((Y_t_cat.shape[1],))
        results.update(dict(C=C, b=b))

    return results

def evaluate_lag_multitrial(
    results,
    z_list,
    x_list,
    u_list,
    L,
    J,
    add_intercept=True,
):
    """
    Evaluate the fitted multi-trial model.

    Computes:
      - R2_z, R2_adj_z, RMSE_z, AIC_z, BIC_z  (for z-equation)
      - R2_x, R2_adj_x, RMSE_x, AIC_x, BIC_x  (for x-equation)

    """

    # Unpack parameters
    A      = np.asarray(results["A"])          # (z_dim, z_dim)
    B_lag  = np.asarray(results["B_lag"])      # (L+1, z_dim, x_dim)
    d_z    = np.asarray(results["d_z"])        # (z_dim,)

    P      = np.asarray(results["P"])          # (x_dim, x_dim)
    K0_lag = np.asarray(results["K0_lag"])     # (J+1, x_dim, z_dim)
    K1_lag = np.asarray(results["K1_lag"])     # (J+1, x_dim, z_dim)
    d_x    = np.asarray(results["d_x"])        # (x_dim,)

    # Accumulate predictions + ground truth
    z_pred_all = []
    z_true_all = []

    x_pred_all = []
    x_true_all = []

    maxlag = max(L, J)

    for z, x, u in zip(z_list, x_list, u_list):
        z = np.asarray(z)         # (T, z_dim)
        x = np.asarray(x)         # (T, x_dim)
        u = np.asarray(u)
        if u.ndim == 1:
            u = u.reshape(-1, 1)  # (T, 1)

        T, z_dim = z.shape
        _, x_dim = x.shape

        # Time window consistent with training
        t0, t1 = maxlag, T - 1
        N = t1 - t0
        if N <= 0:
            continue  # skip pathological short trials

        # Ground-truth next states
        z_t    = z[t0:t1]           # (N, z_dim)
        z_next = z[t0+1:t1+1]       # (N, z_dim)

        x_t    = x[t0:t1]           # (N, x_dim)
        x_next = x[t0+1:t1+1]       # (N, x_dim)

        # Build lags (same as in training)
        X_lags = np.hstack([x[t0-k:t1-k] for k in range(L+1)])   # (N, (L+1)*x_dim)
        Z_lags = np.hstack([z[t0-k:t1-k] for k in range(J+1)])   # (N, (J+1)*z_dim)
        U_lags = np.hstack([u[t0-k:t1-k] for k in range(J+1)])   # (N, (J+1)*1)

        # ---- z-equation prediction ----
        # z_{t+1} = A z_t + sum_k B_k x_{t-k} + d_z
        z_pred = A @ z_t.T                        # (z_dim, N)
        for k in range(L+1):
            x_lag_k = X_lags[:, k*x_dim:(k+1)*x_dim]   # (N, x_dim)
            z_pred += B_lag[k] @ x_lag_k.T             # (z_dim, N)
        z_pred = z_pred.T + d_z                        # (N, z_dim)

        # ---- x-equation prediction ----
        # x_{t+1} = P x_t + sum_k K0_k z_{t-k} + sum_k u_{t-k} K1_k z_{t-k} + d_x
        x_pred = P @ x_t.T                        # (x_dim, N)
        for k in range(J+1):
            z_lag_k = Z_lags[:, k*z_dim:(k+1)*z_dim]   # (N, z_dim)
            u_k     = U_lags[:, [k]]                   # (N, 1)

            x_pred += K0_lag[k] @ z_lag_k.T            # (x_dim, N)
            x_pred += K1_lag[k] @ (u_k * z_lag_k).T    # (x_dim, N)

        x_pred = x_pred.T + d_x                        # (N, x_dim)

        # Collect
        z_pred_all.append(z_pred)
        z_true_all.append(z_next)

        x_pred_all.append(x_pred)
        x_true_all.append(x_next)

    # Stack across trials
    z_pred_all = np.vstack(z_pred_all)    # (N_total, z_dim)
    z_true_all = np.vstack(z_true_all)    # (N_total, z_dim)

    x_pred_all = np.vstack(x_pred_all)    # (N_total, x_dim)
    x_true_all = np.vstack(x_true_all)    # (N_total, x_dim)

    # ---------- Metrics helpers ----------
    def R2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
        return 1.0 - ss_res / ss_tot

    def RMSE(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    # ---------- Count predictors p_z, p_x (matches training design) ----------
    # z-equation design: [Z_t (z_dim), X_lags ((L+1)*x_dim), intercept?]
    z_dim = z_true_all.shape[1]
    x_dim = x_true_all.shape[1]

    p_z_design = z_dim + (L + 1) * x_dim + (1 if add_intercept else 0)

    # x-equation design: [X_t (x_dim), Z_lags ((J+1)*z_dim), UZ_lags ((J+1)*z_dim), intercept?]
    # UZ_lags has same width as Z_lags: (J+1)*z_dim
    p_x_design = x_dim + 2 * (J + 1) * z_dim + (1 if add_intercept else 0)

    # ---------- Compute metrics ----------
    # R² / adj R² / RMSE
    R2_z      = R2(z_true_all, z_pred_all)
    RMSE_z    = RMSE(z_true_all, z_pred_all)

    R2_x      = R2(x_true_all, x_pred_all)
    RMSE_x    = RMSE(x_true_all, x_pred_all)


    metrics = dict(
        R2_z=R2_z,
        RMSE_z=RMSE_z,
        
        R2_x=R2_x,
        RMSE_x=RMSE_x,
    )

    return metrics, x_pred_all, x_true_all

'''
===================
Help function for fitting and evaluating ridge regression normal kinematic only model
===================
'''

def fit_lag_model_ridge_multitrial_noK1(
    z_list, x_list, u_list,
    L, J,
    y_list=None,
    lam_z=1e-2,
    lam_x=1e-2,
    lam_y=1e-2,
    add_intercept=True
):
    """
    Trial-splitted Ridge, NO-K1 version.

    Same z-equation as before.

    x-equation becomes:
      x_{t+1} = P x_t + sum_{k=0}^J K0_k z_{t-k} + d_x
    (No u*z term, no K1 params.)
    """

    # ---------- 1. Build and stack designs over trials ----------
    Phi_z_all = []
    Z_next_all = []
    Phi_x_all = []
    X_next_all = []
    Z_t_all = []
    Y_t_all = [] if y_list is not None else None

    for i, (z, x, u) in enumerate(zip(z_list, x_list, u_list)):
        Phi_z_i, Z_next_i, Phi_x_i, X_next_i, Z_t_i = \
            _build_lag_design_one_trial(z, x, u, L, J, add_intercept=add_intercept)

        Phi_z_all.append(Phi_z_i)
        Z_next_all.append(Z_next_i)
        Phi_x_all.append(Phi_x_i)
        X_next_all.append(X_next_i)
        Z_t_all.append(Z_t_i)

        if y_list is not None:
            y = y_list[i]
            maxlag = max(L, J)
            t0, t1 = maxlag, y.shape[0] - 1
            Y_t_all.append(y[t0:t1])

    Phi_z  = np.vstack(Phi_z_all)      # (N_total, ...)
    Z_next = np.vstack(Z_next_all)     # (N_total, z_dim)
    Phi_x_full = np.vstack(Phi_x_all)  # (N_total, ...)
    X_next = np.vstack(X_next_all)     # (N_total, x_dim)
    Z_t_cat = np.vstack(Z_t_all)       # (N_total, z_dim)

    if y_list is not None:
        Y_t_cat = np.vstack(Y_t_all)   # (N_total, y_dim)

    N_total, z_dim = Z_next.shape
    _, x_dim = X_next.shape

    # ---------- 1b. Drop the (u*z) block from Phi_x ----------
    # Phi_x_full is assumed: [x_t | z_lags | uz_lags | (intercept)]
    base_w = x_dim + (J + 1) * z_dim

    if add_intercept:
        # keep [x_t | z_lags] and the last intercept column
        Phi_x = np.concatenate([Phi_x_full[:, :base_w], Phi_x_full[:, -1:]], axis=1)
    else:
        # keep only [x_t | z_lags]
        Phi_x = Phi_x_full[:, :base_w]

    # ---------- 2. z-ridge ----------
    Theta_z = _ridge_solve(
        Phi_z, Z_next,
        lam=lam_z,
        n_unpenalized=1 if add_intercept else 0
    )

    idx = 0
    A = Theta_z[idx:idx + z_dim, :].T
    idx += z_dim

    B_flat_T = Theta_z[idx:idx + (L + 1) * x_dim, :].T
    idx += (L + 1) * x_dim

    d1 = Theta_z[idx, :] if add_intercept else np.zeros((z_dim,))

    B_lag = B_flat_T.reshape(z_dim, L + 1, x_dim).transpose(1, 0, 2)  # (L+1, z_dim, x_dim)

    # ---------- 3. x-ridge (NO K1) ----------
    Theta_x = _ridge_solve(
        Phi_x, X_next,
        lam=lam_x,
        n_unpenalized=1 if add_intercept else 0
    )

    idx = 0
    P = Theta_x[idx:idx + x_dim, :].T
    idx += x_dim

    K0_flat_T = Theta_x[idx:idx + (J + 1) * z_dim, :].T
    idx += (J+1)*z_dim

    d2 = Theta_x[idx, :] if add_intercept else np.zeros((x_dim,))

    K0_lag = K0_flat_T.reshape(x_dim, J + 1, z_dim).transpose(1, 0, 2)  # (J+1, x_dim, z_dim)

    results = dict(
        A=A, P=P,
        B_lag=B_lag,
        K0_lag=K0_lag,
        d_z=d1,
        d_x=d2,
    )

    # ---------- 4. optional readout (y) ----------
    if y_list is not None:
        Phi_y_parts = [Z_t_cat]
        if add_intercept:
            Phi_y_parts.append(np.ones((N_total, 1)))
        Phi_y = np.concatenate(Phi_y_parts, axis=1)

        Theta_y = _ridge_solve(
            Phi_y, Y_t_cat,
            lam=lam_y,
            n_unpenalized=1 if add_intercept else 0
        )

        if add_intercept:
            C = Theta_y[:-1, :].T
            b = Theta_y[-1, :]
        else:
            C = Theta_y.T
            b = np.zeros((Y_t_cat.shape[1],))

        results.update(dict(C=C, b=b))

    return results


def evaluate_lag_multitrial_noK1(
    results,
    z_list,
    x_list,
    u_list,          # kept for API compatibility, but NOT used
    L,
    J,
    add_intercept=True,
):
    """
    Evaluate the fitted multi-trial model WITHOUT K1.

    z-equation (same):
      z_{t+1} = A z_t + sum_{k=0}^L B_k x_{t-k} + d_z

    x-equation (no K1):
      x_{t+1} = P x_t + sum_{k=0}^J K0_k z_{t-k} + d_x

    """

    A      = np.asarray(results["A"])          # (z_dim, z_dim)
    B_lag  = np.asarray(results["B_lag"])      # (L+1, z_dim, x_dim)
    d_z    = np.asarray(results["d_z"])        # (z_dim,)

    P      = np.asarray(results["P"])          # (x_dim, x_dim)
    K0_lag = np.asarray(results["K0_lag"])     # (J+1, x_dim, z_dim)
    d_x    = np.asarray(results["d_x"])        # (x_dim,)

    z_pred_all, z_true_all = [], []
    x_pred_all, x_true_all = [], []

    maxlag = max(L, J)

    for z, x, u in zip(z_list, x_list, u_list):
        z = np.asarray(z)         # (T, z_dim)
        x = np.asarray(x)         # (T, x_dim)

        T, z_dim = z.shape
        _, x_dim = x.shape

        t0, t1 = maxlag, T - 1
        N = t1 - t0
        if N <= 0:
            continue 

        # Ground-truth next states
        z_t    = z[t0:t1]           # (N, z_dim)
        z_next = z[t0+1:t1+1]       # (N, z_dim)

        x_t    = x[t0:t1]           # (N, x_dim)
        x_next = x[t0+1:t1+1]       # (N, x_dim)

        # Build lags (same as in training)
        X_lags = np.hstack([x[t0-k:t1-k] for k in range(L+1)])   # (N, (L+1)*x_dim)
        Z_lags = np.hstack([z[t0-k:t1-k] for k in range(J+1)])   # (N, (J+1)*z_dim)

        # ---- z-equation prediction ----
        # z_{t+1} = A z_t + sum_k B_k x_{t-k} + d_z
        z_pred = A @ z_t.T                        # (z_dim, N)
        for k in range(L+1):
            x_lag_k = X_lags[:, k*x_dim:(k+1)*x_dim]   # (N, x_dim)
            z_pred += B_lag[k] @ x_lag_k.T             # (z_dim, N)
        z_pred = z_pred.T + d_z                        # (N, z_dim)

        # ---- x-equation prediction (NO K1) ----
        # x_{t+1} = P x_t + sum_k K0_k z_{t-k} + d_x
        x_pred = P @ x_t.T                        # (x_dim, N)
        for k in range(J+1):
            z_lag_k = Z_lags[:, k*z_dim:(k+1)*z_dim]   # (N, z_dim)
            x_pred += K0_lag[k] @ z_lag_k.T            # (x_dim, N)
        x_pred = x_pred.T + d_x                        # (N, x_dim)

        z_pred_all.append(z_pred)
        z_true_all.append(z_next)
        x_pred_all.append(x_pred)
        x_true_all.append(x_next)

    # Stack across trials
    z_pred_all = np.vstack(z_pred_all)
    z_true_all = np.vstack(z_true_all)
    x_pred_all = np.vstack(x_pred_all)
    x_true_all = np.vstack(x_true_all)

    # ---------- Metrics helpers ----------
    def R2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
        return 1.0 - ss_res / ss_tot

    def RMSE(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    # ---------- Compute metrics ----------
    R2_z     = R2(z_true_all, z_pred_all)
    RMSE_z   = RMSE(z_true_all, z_pred_all)

    R2_x     = R2(x_true_all, x_pred_all)
    RMSE_x   = RMSE(x_true_all, x_pred_all)

    return dict(
        R2_z=R2_z, RMSE_z=RMSE_z, 
        R2_x=R2_x, RMSE_x=RMSE_x, 
    ), x_pred_all, x_true_all

'''
===================
Help function for fitting and evaluating ridge regression intrinsic only model
===================
'''

def fit_lag_model_ridge_multitrial_Ponly(
    z_list, x_list, u_list,
    L, J,
    y_list=None,
    lam_z=1e-2,
    lam_x=1e-2,
    lam_y=1e-2,
    add_intercept=True
):
    """
    Trial-splitted Ridge.

    z-equation: SAME as before
      z_{t+1} = A z_t + sum_{k=0}^L B_k x_{t-k} + d_z

    x-equation: P-ONLY
      x_{t+1} = P x_t + d_x
    """

    # ---------- 1. Build and stack designs over trials ----------
    Phi_z_all, Z_next_all = [], []
    Phi_x_all, X_next_all = [], []
    Z_t_all = []
    Y_t_all = [] if y_list is not None else None

    for i, (z, x, u) in enumerate(zip(z_list, x_list, u_list)):
        Phi_z_i, Z_next_i, Phi_x_i, X_next_i, Z_t_i = \
            _build_lag_design_one_trial(z, x, u, L, J, add_intercept=add_intercept)

        Phi_z_all.append(Phi_z_i)
        Z_next_all.append(Z_next_i)
        Phi_x_all.append(Phi_x_i)
        X_next_all.append(X_next_i)
        Z_t_all.append(Z_t_i)

        if y_list is not None:
            y = y_list[i]
            maxlag = max(L, J)
            t0, t1 = maxlag, y.shape[0] - 1
            Y_t_all.append(y[t0:t1])

    Phi_z   = np.vstack(Phi_z_all)       # (N_total, ...)
    Z_next  = np.vstack(Z_next_all)      # (N_total, z_dim)
    Phi_x_full = np.vstack(Phi_x_all)    # (N_total, ...)
    X_next  = np.vstack(X_next_all)      # (N_total, x_dim)
    Z_t_cat = np.vstack(Z_t_all)         # (N_total, z_dim)

    if y_list is not None:
        Y_t_cat = np.vstack(Y_t_all)     # (N_total, y_dim)

    N_total, z_dim = Z_next.shape
    _, x_dim = X_next.shape

    # ---------- 1b. Keep ONLY [x_t] (+ intercept) for Phi_x ----------
    if add_intercept:
        # [x_t] are lagst x_dim cols; intercept is last col
        Phi_x = np.concatenate([Phi_x_full[:, :x_dim], Phi_x_full[:, -1:]], axis=1)
    else:
        Phi_x = Phi_x_full[:, :x_dim]

    # ---------- 2. z-ridge (unchanged) ----------
    Theta_z = _ridge_solve(
        Phi_z, Z_next,
        lam=lam_z,
        n_unpenalized=1 if add_intercept else 0
    )

    idx = 0
    A = Theta_z[idx:idx + z_dim, :].T
    idx += z_dim

    B_flat_T = Theta_z[idx:idx + (L + 1) * x_dim, :].T
    idx += (L + 1) * x_dim

    d_z = Theta_z[idx, :] if add_intercept else np.zeros((z_dim,))
    B_lag = B_flat_T.reshape(z_dim, L + 1, x_dim).transpose(1, 0, 2)  # (L+1, z_dim, x_dim)

    # ---------- 3. x-ridge (P-only) ----------
    Theta_x = _ridge_solve(
        Phi_x, X_next,
        lam=lam_x,
        n_unpenalized=1 if add_intercept else 0
    )

    idx = 0
    P = Theta_x[idx:idx + x_dim, :].T
    idx += x_dim

    d_x = Theta_x[idx, :] if add_intercept else np.zeros((x_dim,))

    results = dict(
        A=A, B_lag=B_lag, d_z=d_z,
        P=P, d_x=d_x,
    )

    # ---------- 4. optional readout (y) ----------
    if y_list is not None:
        Phi_y_parts = [Z_t_cat]
        if add_intercept:
            Phi_y_parts.append(np.ones((N_total, 1)))
        Phi_y = np.concatenate(Phi_y_parts, axis=1)

        Theta_y = _ridge_solve(
            Phi_y, Y_t_cat,
            lam=lam_y,
            n_unpenalized=1 if add_intercept else 0
        )

        if add_intercept:
            C = Theta_y[:-1, :].T
            b = Theta_y[-1, :]
        else:
            C = Theta_y.T
            b = np.zeros((Y_t_cat.shape[1],))

        results.update(dict(C=C, b=b))

    return results


def evaluate_lag_multitrial_Ponly(
    results,
    z_list,
    x_list,
    u_list,          # kept for API compatibility, not used
    L,
    J,
    add_intercept=True,
):
    """
    Evaluate model with P-only x-equation:

      z_{t+1} = A z_t + sum_{k=0}^L B_k x_{t-k} + d_z
      x_{t+1} = P x_t + d_x

    Returns R2/RMSE/AIC/BIC for both equations.
    """

    # Unpack parameters
    A      = np.asarray(results["A"])
    B_lag  = np.asarray(results["B_lag"])
    d_z    = np.asarray(results["d_z"])

    P      = np.asarray(results["P"])
    d_x    = np.asarray(results["d_x"])

    z_pred_all, z_true_all = [], []
    x_pred_all, x_true_all = [], []

    maxlag = max(L, J)

    for z, x, _u in zip(z_list, x_list, u_list):
        z = np.asarray(z)
        x = np.asarray(x)

        T, z_dim = z.shape
        _, x_dim = x.shape

        t0, t1 = maxlag, T - 1
        N = t1 - t0
        if N <= 0:
            continue

        z_t    = z[t0:t1]
        z_next = z[t0+1:t1+1]

        x_t    = x[t0:t1]
        x_next = x[t0+1:t1+1]

        X_lags = np.hstack([x[t0-k:t1-k] for k in range(L+1)])   # (N, (L+1)*x_dim)

        # z prediction
        z_pred = A @ z_t.T
        for k in range(L+1):
            x_lag_k = X_lags[:, k*x_dim:(k+1)*x_dim]
            z_pred += B_lag[k] @ x_lag_k.T
        z_pred = z_pred.T + d_z

        # x prediction (P-only)
        x_pred = (P @ x_t.T).T + d_x

        z_pred_all.append(z_pred)
        z_true_all.append(z_next)
        x_pred_all.append(x_pred)
        x_true_all.append(x_next)

    z_pred_all = np.vstack(z_pred_all)
    z_true_all = np.vstack(z_true_all)
    x_pred_all = np.vstack(x_pred_all)
    x_true_all = np.vstack(x_true_all)

    # ----- metrics helpers -----
    def R2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
        return 1.0 - ss_res / ss_tot

    def RMSE(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


    R2_z = R2(z_true_all, z_pred_all)
    RMSE_z = RMSE(z_true_all, z_pred_all)

    R2_x = R2(x_true_all, x_pred_all)
    RMSE_x = RMSE(x_true_all, x_pred_all)

    return dict(
        R2_z=R2_z, RMSE_z=RMSE_z,
        R2_x=R2_x, RMSE_x=RMSE_x,
    ), x_pred_all, x_true_all


'''
===================
Helper function for splitting components in model prediction
===================
'''
def pred_lag_multitrial(
    results,
    z_list,
    x_list,
    u_list,
    L,
    J,
):
    """
    Multi-trial prediction for the x-equation.

    Accepts:
      z: (B,T,z_dim) or (T,z_dim)
      x: (B,T,x_dim) or (T,x_dim)
      u: (B,T) or (B,T,1) or (T,) or (T,1)

    Returns dict with arrays shaped:
      (B, N, x_dim) for all per-time predictions,
      plus t0, t1, N, maxlag for bookkeeping.
    """

    P      = np.asarray(results["P"])          # (x_dim, x_dim)
    K0_lag = np.asarray(results["K0_lag"])     # (J+1, x_dim, z_dim)
    K1_lag = np.asarray(results["K1_lag"])     # (J+1, x_dim, z_dim)
    d_x    = np.asarray(results["d_x"])        # (x_dim,)

    maxlag = max(L, J)

    # ---------- normalize shapes to (B, T, ...) ----------
    z = np.asarray(z_list)
    x = np.asarray(x_list)
    u = np.asarray(u_list)

    # z
    if z.ndim == 2:  # (T, z_dim)
        z = z[None, ...]
    elif z.ndim != 3:
        raise ValueError(f"z must be (T,z_dim) or (B,T,z_dim). Got {z.shape}")

    # x
    if x.ndim == 2:  # (T, x_dim)
        x = x[None, ...]
    elif x.ndim != 3:
        raise ValueError(f"x must be (T,x_dim) or (B,T,x_dim). Got {x.shape}")

    # u
    if u.ndim == 1:          # (T,)
        u = u[None, :, None] # (1,T,1)
    elif u.ndim == 2:        # (B,T) or (T,1)
        if u.shape[1] == 1 and u.shape[0] == z.shape[1]:  # (T,1)
            u = u[None, ...]  # (1,T,1)
        else:
            u = u[:, :, None] # (B,T,1)
    elif u.ndim == 3:        # (B,T,1) or (1,T,1)
        pass
    else:
        raise ValueError(f"u must be (T,), (T,1), (B,T), or (B,T,1). Got {u.shape}")

    B, T, z_dim = z.shape
    Bx, Tx, x_dim = x.shape
    Bu, Tu, u_dim = u.shape

    if (Bx != B) or (Tx != T):
        raise ValueError(f"z and x must match in (B,T). Got z {z.shape}, x {x.shape}")
    if (Bu != B) or (Tu != T):
        raise ValueError(f"u must match z/x in (B,T). Got u {u.shape}, z {z.shape}")
    if u_dim != 1:
        raise ValueError(f"u last dim must be 1. Got u_dim={u_dim}")

    # ---------- time window ----------
    t0, t1 = maxlag, T - 1  # predict x_{t+1} for t in [t0, t1-1]
    N = t1 - t0
    if N <= 0:
        raise ValueError(f"T={T} too short for maxlag={maxlag}. Need T >= maxlag+2.")

    x_t = x[:, t0:t1, :]         # (B,N,x_dim)

    # ---------- base: P x_t + d_x ----------
    # pred_xP: (B,N,x_dim)
    pred_xP = x_t @ P.T + d_x[None, None, :]

    # ---------- add K0 and gated K1 terms ----------
    pred_x0 = np.zeros_like(pred_xP)  # (B,N,x_dim)
    pred_x1 = np.zeros_like(pred_xP)  # (B,N,x_dim)

    for k in range(J + 1):
        z_lag_k = z[:, (t0 - k):(t1 - k), :]  # (B,N,z_dim)
        u_k     = u[:, (t0 - k):(t1 - k), :]  # (B,N,1)

        # K0: (x_dim,z_dim); z_lag_k: (B,N,z_dim) -> (B,N,x_dim)
        pred_x0 += z_lag_k @ K0_lag[k].T

        # gated: u_k * z_lag_k -> (B,N,z_dim)
        pred_x1 += (u_k * z_lag_k) @ K1_lag[k].T

    pred_x = pred_xP + pred_x0 + pred_x1

    res = dict(
        pred_x=pred_x,                              # (B,N,x_dim)
        pred_x_no_control=pred_xP,                  # (B,N,x_dim)
        pred_x01=(pred_x0 + pred_x1),               # (B,N,x_dim)
        pred_x0=pred_x0,                            # (B,N,x_dim)
        pred_x1=pred_x1,                            # (B,N,x_dim)
        t0=t0, t1=t1, N=N, maxlag=maxlag,
    )
    return res


'''
===================
Helper function for fitting task plane
===================
'''

def fit_trk_plane(traj, orient_z_positive=True):
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

def compute_rmse(traj, reg):
    X = traj[:, :2]
    z = traj[:, 2]
    z_pred = reg.predict(X)
    residuals = z - z_pred
    return np.sqrt(np.mean(residuals**2))

def project_onto_plane(points, normal, d):
    """
    Project points onto a plane defined by its normal vector and offset d.
    Plane equation: ax + by + cz + d = 0
    """
    normal = normal / np.linalg.norm(normal)
    distances = (points @ normal[:3] + d)
    projections = points - np.outer(distances, normal)
    return projections


def rotate_points_to_z(points, normal):
    normal = normal / np.linalg.norm(normal)
    z_axis = np.array([0,0,1])
    rotation_axis = np.cross(normal, z_axis)
    axis_norm = np.linalg.norm(rotation_axis)
    
    if axis_norm < 1e-8:
        R = np.eye(3)
    else:
        rotation_axis /= axis_norm
        theta = np.arccos(np.clip(np.dot(normal, z_axis), -1, 1))
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])
        R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
    
    rotated_points = points @ R.T
    return rotated_points, R





'''
===================
Helper function for cross validation
===================
'''

def cv_z_lag_and_lambda(
    z_trials,
    x_trials,
    u_trials,
    L_list,
    lambdas_z,
    J_fixed,
    lambda_x_fixed,
    K=5,
    selection_metric="R2_z",   # optimize z-side metric
    random_state=0,
    verbose=True,
):
    """
    Cross-validate (L, lambda_z) for the z-equation, holding (J, lambda_x) fixed.
    """

    z_trials = np.asarray(z_trials)
    x_trials = np.asarray(x_trials)
    u_trials = np.asarray(u_trials)

    L_list = list(L_list)
    nL = len(L_list)
    lambdas_z = np.array(lambdas_z)
    nLam = len(lambdas_z)

    # grids for inspection
    full_score_grid = np.full((nL, nLam), np.nan)
    best_lambda_z_per_L = np.full(nL, np.nan)
    best_score_per_L = np.full(nL, np.nan)

    all_results_z = []
    best_config_z = None

    kf = KFold(n_splits=K, shuffle=True, random_state=random_state)

    # We'll detect metric names on lagst run
    metric_names = None

    for iL, L in enumerate(L_list):
        if verbose:
            print("\n" + "=" * 60)
            print(f"Z-side CV for L={L} (J fixed = {J_fixed})")
            print("=" * 60)

        for il, lam_z in enumerate(lambdas_z):
            # accumulate across folds
            fold_scores = []

            # for all metrics if you want, but we only store selection_metric in grid
            # we can also keep raw if needed
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(z_trials)):
                z_train, z_val = z_trials[train_idx], z_trials[val_idx]
                x_train, x_val = x_trials[train_idx], x_trials[val_idx]
                u_train, u_val = u_trials[train_idx], u_trials[val_idx]

                # fit model with given hyperparams
                results = fit_lag_model_ridge_multitrial(
                    z_train, x_train, u_train,
                    L=L,
                    J=J_fixed,
                    lam_z=lam_z,
                    lam_x=lambda_x_fixed,
                )

                # evaluate on validation set
                metrics_val, x_all, x_pred = evaluate_lag_multitrial(
                    results, z_val, x_val, u_val, L=L, J=J_fixed
                )

                if metric_names is None:
                    metric_names = sorted(metrics_val.keys())
                    if selection_metric not in metric_names:
                        raise ValueError(
                            f"selection_metric '{selection_metric}' not in metrics {metric_names}"
                        )

                fold_scores.append(metrics_val[selection_metric])

            mean_val_score = np.mean(fold_scores)
            full_score_grid[iL, il] = mean_val_score

            if verbose:
                print(
                    f"L={L}, lambda_z={lam_z:.3e} -> "
                    f"val {selection_metric}={mean_val_score:.4f}"
                )

        # for this L, choose best lambda_z
        row = full_score_grid[iL, :]
        best_idx = np.nanargmax(row)
        best_lambda_z = float(lambdas_z[best_idx])
        best_score = float(row[best_idx])

        best_lambda_z_per_L[iL] = best_lambda_z
        best_score_per_L[iL] = best_score

        if verbose:
            print(
                f"--> For L={L}: best lambda_z={best_lambda_z:.3e}, "
                f"val {selection_metric}={best_score:.4f}"
            )

        # store L-level result
        all_results_z.append({
            'L': L,
            'best_lambda_z': best_lambda_z,
            'best_val_score': best_score,
            'scores_for_lambdas': row.copy(),
        })

        # update global best over L
        if (best_config_z is None) or (best_score > best_config_z['val_score']):
            best_config_z = {
                'L': L,
                'lambda_z': best_lambda_z,
                'val_score': best_score,
            }

    if verbose:
        print("\n=== Z-side CV complete ===")
        print(
            f"Best L={best_config_z['L']}, "
            f"lambda_z={best_config_z['lambda_z']:.3e}, "
            f"val {selection_metric}={best_config_z['val_score']:.4f}"
        )

    return best_config_z, all_results_z, best_lambda_z_per_L, best_score_per_L, full_score_grid

def cv_x_lag_and_lambda(
    z_trials,
    x_trials,
    u_trials,
    J_list,
    lambdas_x,
    L_fixed,
    lambda_z_fixed,
    K=5,
    selection_metric="R2_x",   # optimize x-side metric
    random_state=0,
    verbose=True,
):
    """
    Cross-validate (J, lambda_x) for the x-equation, holding (L, lambda_z) fixed.
    """

    z_trials = np.asarray(z_trials)
    x_trials = np.asarray(x_trials)
    u_trials = np.asarray(u_trials)

    J_list = list(J_list)
    nJ = len(J_list)
    lambdas_x = np.array(lambdas_x)
    nLam = len(lambdas_x)

    full_score_grid = np.full((nJ, nLam), np.nan)
    best_lambda_x_per_J = np.full(nJ, np.nan)
    best_score_per_J = np.full(nJ, np.nan)

    all_results_x = []
    best_config_x = None

    kf = KFold(n_splits=K, shuffle=True, random_state=random_state)
    metric_names = None

    for iJ, J in enumerate(J_list):
        if verbose:
            print("\n" + "=" * 60)
            print(f"X-side CV for J={J} (L fixed = {L_fixed})")
            print("=" * 60)

        for il, lam_x in enumerate(lambdas_x):
            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(z_trials)):
                z_train, z_val = z_trials[train_idx], z_trials[val_idx]
                x_train, x_val = x_trials[train_idx], x_trials[val_idx]
                u_train, u_val = u_trials[train_idx], u_trials[val_idx]

                results = fit_lag_model_ridge_multitrial(
                    z_train, x_train, u_train,
                    L=L_fixed,
                    J=J,
                    lam_z=lambda_z_fixed,
                    lam_x=lam_x,
                )

                metrics_val, x_all, x_pred = evaluate_lag_multitrial(
                    results, z_val, x_val, u_val, L=L_fixed, J=J
                )

                if metric_names is None:
                    metric_names = sorted(metrics_val.keys())
                    if selection_metric not in metric_names:
                        raise ValueError(
                            f"selection_metric '{selection_metric}' not in metrics {metric_names}"
                        )

                fold_scores.append(metrics_val[selection_metric])

            mean_val_score = np.mean(fold_scores)
            full_score_grid[iJ, il] = mean_val_score

            if verbose:
                print(
                    f"J={J}, lambda_x={lam_x:.3e} -> "
                    f"val {selection_metric}={mean_val_score:.4f}"
                )

        # choose best lambda_x for this J
        row = full_score_grid[iJ, :]
        best_idx = np.nanargmax(row)
        best_lambda_x = float(lambdas_x[best_idx])
        best_score = float(row[best_idx])

        best_lambda_x_per_J[iJ] = best_lambda_x
        best_score_per_J[iJ] = best_score

        if verbose:
            print(
                f"--> For J={J}: best lambda_x={best_lambda_x:.3e}, "
                f"val {selection_metric}={best_score:.4f}"
            )

        all_results_x.append({
            'J': J,
            'best_lambda_x': best_lambda_x,
            'best_val_score': best_score,
            'scores_for_lambdas': row.copy(),
        })

        if (best_config_x is None) or (best_score > best_config_x['val_score']):
            best_config_x = {
                'J': J,
                'lambda_x': best_lambda_x,
                'val_score': best_score,
            }

    if verbose:
        print("\n=== X-side CV complete ===")
        print(
            f"Best J={best_config_x['J']}, "
            f"lambda_x={best_config_x['lambda_x']:.3e}, "
            f"val {selection_metric}={best_config_x['val_score']:.4f}"
        )

    return best_config_x, all_results_x, best_lambda_x_per_J, best_score_per_J, full_score_grid
