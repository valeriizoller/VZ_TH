import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import jarque_bera, chi2
from statsmodels.stats.diagnostic import acorr_ljungbox

def multivariate_jarque_bera(residuals: np.ndarray):
    """
    Perform Jarque-Bera test for multivariate data.
    
    Parameters:
    - residuals (np.ndarray): NxT array where each column is a time series observation.

    Returns:
    - jb_stat (float): The combined Jarque-Bera statistic for all time series.
    - p_value (float): The p-value corresponding to the JB statistic.
    """
    N, T = residuals.shape
    jb_stats = []
    p_values = []
    
    for i in range(T):
        jb_stat, p_value = jarque_bera(residuals[:, i])
        jb_stats.append(jb_stat)
        p_values.append(p_value)
    
    # Combine the JB statistics by summing them up
    combined_jb_stat = np.sum(jb_stats)
    # Degrees of freedom is 2*T because JB test has 2 degrees of freedom per time series
    combined_p_value = 1 - chi2.cdf(combined_jb_stat, df=2*T)
    
    return combined_jb_stat, combined_p_value

def multivariate_ljung_box(residuals: np.ndarray, lags=10):
    """
    Perform Ljung-Box test for autocorrelation for multivariate data.
    
    Parameters:
    - residuals (np.ndarray): NxT array where each column is a time series observation.
    - lags (int): Number of lags to test for autocorrelation.
    
    Returns:
    - lb_stat (float): The combined Ljung-Box statistic for all time series.
    - p_value (float): The p-value corresponding to the LB statistic.
    """
    N, T = residuals.shape
    lb_stats = []
    p_values = []
    
    for i in range(T):
        lb_test = acorr_ljungbox(residuals[:, i], lags=[lags], return_df=False)
        lb_stat = lb_test[0]
        p_value = lb_test[1]
        
        lb_stats.append(lb_stat)
        p_values.append(p_value)
    
    # Combine the LB statistics by summing them up
    combined_lb_stat = np.sum(lb_stats)
    # Degrees of freedom is T because LB test has 1 degree of freedom per time series per lag
    combined_p_value = 1 - chi2.cdf(combined_lb_stat, df=T)
    
    return combined_lb_stat, combined_p_value

def calculate_acf(time_series, nlags, jumps):
    """
    Calculate the Autocorrelation Function (ACF) for a time series.

    Parameters:
    - time_series: array-like, the time series data
    - nlags: int, the number of lags to compute

    Returns:
    - acf_values: numpy array, the ACF values up to the specified lag
    """
    # Compute ACF values
    acf_values = acf(time_series, nlags=nlags*jumps, fft=True)
    return acf_values[::jumps]

def calculate_pacf(time_series, nlags, jumps,method='ols-adjusted'):
    """
    Calculate the Partial Autocorrelation Function (PACF) for a time series.

    Parameters:
    - time_series: array-like, the time series data
    - nlags: int, the number of lags to compute

    Returns:
    - pacf_values: numpy array, the PACF values up to the specified lag
    """
    # Compute PACF values
    pacf_values = pacf(time_series, nlags=nlags*jumps, method=method)
    return pacf_values[::jumps]

def multivariate_jarque_bera(residuals: np.ndarray):
    """
    Perform Jarque-Bera test for multivariate data.
    
    Parameters:
    - residuals (np.ndarray): NxT array where each column is a time series observation.

    Returns:
    - jb_stat (float): The combined Jarque-Bera statistic for all time series.
    - p_value (float): The p-value corresponding to the JB statistic.
    """
    N, T = residuals.shape
    jb_stats = []
    p_values = []
    
    for i in range(T):
        jb_stat, p_value = jarque_bera(residuals[:, i])
        jb_stats.append(jb_stat)
        p_values.append(p_value)
    
    # Combine the JB statistics by summing them up
    combined_jb_stat = np.sum(jb_stats)
    # Degrees of freedom is 2*T because JB test has 2 degrees of freedom per time series
    combined_p_value = 1 - chi2.cdf(combined_jb_stat, df=2*T)
    
    return combined_jb_stat, combined_p_value

def multivariate_ljung_box(residuals: np.ndarray, lags=10):
    """
    Perform Ljung-Box test for autocorrelation for multivariate data.
    
    Parameters:
    - residuals (np.ndarray): NxT array where each column is a time series observation.
    - lags (int): Number of lags to test for autocorrelation.
    
    Returns:
    - lb_stat (float): The combined Ljung-Box statistic for all time series.
    - p_value (float): The p-value corresponding to the LB statistic.
    """
    N, T = residuals.shape
    lb_stats = []
    p_values = []
    
    for i in range(T):
        lb_test = acorr_ljungbox(residuals[:, i], lags=[lags], return_df=False)
        lb_stat = lb_test[0]
        p_value = lb_test[1]
        
        lb_stats.append(lb_stat)
        p_values.append(p_value)
    
    # Combine the LB statistics by summing them up
    combined_lb_stat = np.sum(lb_stats)
    # Degrees of freedom is T because LB test has 1 degree of freedom per time series per lag
    combined_p_value = 1 - chi2.cdf(combined_lb_stat, df=T)
    
    return combined_lb_stat, combined_p_value


