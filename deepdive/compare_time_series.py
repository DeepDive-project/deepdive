import numpy as np
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import scipy.stats
np.set_printoptions(suppress=True, precision=3)
import matplotlib
import matplotlib.pyplot as plt
from collections.abc import Iterable
import pandas as pd

np.random.seed(123)


def generate_time_series(n_steps, x0=0):
    x = [x0]
    for i in range(n_steps - 1):
        x.append(np.random.normal() + x[i])
    return np.array(x)


def generate_correlates(time_series, k=4, err=0.2):
    coef = np.ones(k) * np.random.uniform(0.5, 3, k)
    rel_noise = np.random.normal(0, err, (len(time_series), k))
    covariates = np.einsum('a,b->ab', Y, coef)
    return coef, rel_noise, covariates + covariates * rel_noise


def calc_time_series_diff(x, y, x_compare=None):
    """
    Time series are expected to be 1D arrays
    :param x: predicted time series (e.g. by DeepDive or SQS) <- can include NAs!
    :param y: true time series as simulated <- can include padding (0s)
    :param x_compare: another predicted time series (typically from SQS) <- can include NAs
    :return: a dictionary with stats comparing x with y excluding:
            1. all bins with 0 diversity in y
            2. all bins with NA in x
            3. all bins with NA in x_compare
    """

    actual_trajectory_indx = np.where(y > 0)[0]
    x_non_na_indx = np.where(np.isfinite(x))[0]
    x_non_zero_indx = np.where(x > 0)[0]
    x_non_na_indx = np.append(x_non_na_indx, x_non_zero_indx)
    keep_indx = np.intersect1d(actual_trajectory_indx, x_non_na_indx)
    # print(len(keep_indx), len(actual_trajectory_indx))
    prediction_fraction = len(keep_indx) / len(actual_trajectory_indx)

    if isinstance(x_compare, Iterable):
        x_compare_non_na_indx = np.where(np.isfinite(x_compare))[0]
        keep_indx = np.intersect1d(keep_indx, x_compare_non_na_indx)
    keep_indx = np.unique(keep_indx)

    if len(keep_indx) > 1:

        x_red = x[keep_indx]
        y_red = y[keep_indx]
        # cumulative distribution
        x_cum = np.cumsum(x_red)
        y_cum = np.cumsum(y_red)
        # derivative
        x_dev = np.diff(x_red)
        y_dev = np.diff(y_red)

        # linear models
        #  slope, intercept, r_value, p_value, std_err
        lm = scipy.stats.linregress(y_red, x_red)
        lm_cum = scipy.stats.linregress(y_cum, x_cum)
        lm_derivative = scipy.stats.linregress(y_dev, x_dev)

        # KL divergence (only makes sense for derivative)
        calc_kld = tf.keras.losses.KLDivergence()
        kl_derivative = calc_kld(y_dev, x_dev).numpy()

        # Kolmogorov-Smirnov test
        from scipy import stats
        ks = stats.kstest(x_red, y_red)[0]
        ks_cum = stats.kstest(x_cum, y_cum)[0]
        ks_derivative = stats.kstest(x_dev, y_dev)[0]

        # mape (doesn't work for derivative)
        calc_mape = tf.keras.losses.MeanAbsolutePercentageError()
        mape = calc_mape(y_red, x_red).numpy()
        mape_cum = calc_mape(y_cum, x_cum).numpy()

        # mse
        calc_mse = tf.keras.losses.MeanSquaredError()
        mse = calc_mse(x_red, y_red).numpy()
        mse_cum = calc_mse(x_cum, y_cum).numpy()
        mse_derivative = calc_mse(x_dev, y_dev).numpy()

        # relative mse
        x_red /= np.max(x_red)
        y_red /= np.max(y_red)
        relative_mse = calc_mse(x_red, y_red).numpy()

        res = {
            "prediction_fraction": prediction_fraction,
            "lm_slope": lm[0],
            "lm_cum_slope": lm_cum[0],
            "lm_derivative_slope": lm_derivative[0],
            "lm_r2": lm[2]**2,
            "lm_cum_r2": lm_cum[2]**2,
            "lm_derivative_r2": lm_derivative[2]**2,
            "kl_derivative": kl_derivative,
            "ks": ks,
            "ks_cum": ks_cum,
            "ks_derivative": ks_derivative,
            "mape": mape,
            "mape_cum": mape_cum,
            "mse": mse,
            "mse_cum": mse_cum,
            "mse_derivative": mse_derivative,
            "relative_mse": relative_mse
        }
    else:
        res = {
            "prediction_fraction": prediction_fraction,
            "lm_slope": np.nan,
            "lm_cum_slope": np.nan,
            "lm_derivative_slope": np.nan,
            "lm_r2": np.nan,
            "lm_cum_r2": np.nan,
            "lm_derivative_r2": np.nan,
            "kl_derivative": np.nan,
            "ks": np.nan,
            "ks_cum": np.nan,
            "ks_derivative": np.nan,
            "mape": np.nan,
            "mape_cum": np.nan,
            "mse": np.nan,
            "mse_cum": np.nan,
            "mse_derivative": np.nan,
            "relative_mse": np.nan
        }
    # res.to_csv('t_series_diff.csv', index=False)
    return res


def calc_time_series_diff2D(X, Y, X_compare=None):
    """
    :param X: 2D or 3D array with predicted trajectories <- can contain NAs and 0s
    :param Y: 2D array with true trajectories <- can contain 0s
    :param X_compare: 2D or 3D array with predicted trajectories
                      for comparison  <- can contain NAs and 0s
    :return: pandas dataframe with the stats for all rows in X
    """
    if len(X.shape) == 3:
        X = X[:, :, 0]

    if X_compare is not None:
        if len(X_compare.shape) == 3:
            X_compare = X_compare[:, :, 0]

    res_l = list()
    for i in range(Y.shape[0]):
        if i % 100 == 0:
            print(i)
        y = Y[i, :]
        x = X[i, :]
        if X_compare is not None:
            x_compare = X_compare[i, :]
        else:
            x_compare = None
        res = calc_time_series_diff(x, y, x_compare=x_compare)
        res_l.append(res)

    df = pd.DataFrame(res_l)
    # df.to_csv('t_series_diff_2d.csv', index=False)
    return df

