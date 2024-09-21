import numpy as np
import pandas as pd
from numpy import linalg as lg
import matplotlib.pyplot as plt
from .Frequency_decomposition import *
from .factors_construction import *
from .utils_algorithm import *
from scipy.stats import chi2, norm

from sklearn.linear_model import LinearRegression


def standardize_time_series(time_series: np.ndarray)->dict:
    mean=np.nanmean(time_series, axis=1, keepdims=True)

    standard_deviation=np.nanstd(time_series, axis=1, keepdims=True)

    standardized_time_series=(time_series-mean)/standard_deviation

    return_dict={
        "Mean": mean,
        "Std": standard_deviation,
        "Standardized_Time_Series" : standardized_time_series,
        "Non_Standardized_Time_Series" : time_series

    }

    return return_dict

def derestandardize_time_series(Standardized_time_series: np.ndarray, mean= np.ndarray, standard_deviation=np.ndarray)->np.ndarray:

    time_series=Standardized_time_series*standard_deviation+mean

    return time_series


def YW_residuals(time_series: np.ndarray, num_lags, overlap=1, force_stability=True):
    """
    Estimate a diagonal VAR model using Yule-Walker equations and compute residuals for Sigma.
    
    Parameters:
    - time_series: NxT numpy array of time series data (N series, T time points).
    - num_lags: Number of lags for the VAR model.
    - overlay: Overlay parameter, specifying the step size for lags.
    
    Returns:
    - results: Dictionary with:
        - "Phi_0": Nx1 numpy array of fixed coefficients (means).
        - "Parameters": List of NxN numpy arrays, the diagonal coefficient matrices for each lag.
        - "Sigma": NxN numpy array, the diagonal covariance matrix of the residuals.
    """
    N, T = time_series.shape
    
    # Initialize the autocovariance matrices
    autocovariances = [np.zeros((N, N)) for _ in range(num_lags + 1)]
    

    mu_mean=np.nanmean(time_series, axis=1, keepdims=True)

    for current_lag in range(num_lags+1):
        for variable_index in range(N):

            valid_start = np.where(~np.isnan(time_series[variable_index]))[0][0]
            valid_data = time_series[variable_index, valid_start:]

            current_T = len(valid_data)

            cov_list =  [np.cov(valid_data[j: current_T-current_lag*overlap: overlap], valid_data[j+current_lag*overlap:: overlap])[0,1] for j in range(np.min([overlap,current_T-(current_lag+1)*overlap]))]
            autocovariances[current_lag][variable_index, variable_index] = np.mean(cov_list, axis=0)


    main_diag=np.diag(autocovariances[0]).copy()
    main_diag[main_diag == 0]=1
    autocovariance_main_diag = np.diag(main_diag)
    # Prepare the left-hand side matrix (R)
    Gamma_0 = np.zeros((N * num_lags, N * num_lags))
    for i in range(num_lags):
        for j in range(num_lags):
            if i == j:
                Gamma_0[i * N:(i + 1) * N, j * N:(j + 1) * N] = autocovariance_main_diag
            elif i > j:
                Gamma_0[i * N:(i + 1) * N, j * N:(j + 1) * N] = autocovariances[i - j]
            else:
                Gamma_0[i * N:(i + 1) * N, j * N:(j + 1) * N] = autocovariances[j - i].T

    

    # Prepare the right-hand side matrix (P)
    Gamma_1 = np.zeros((N * num_lags, N))
    for i in range(num_lags):
        Gamma_1[i * N:(i + 1) * N, :] = autocovariances[i + 1]

    coefficients = np.linalg.solve(Gamma_0, Gamma_1).T

    # Extract the coefficients for the requested lags
    parameters= [coefficients[:, N*(i-1):N*(i)] for i in range(1, num_lags+1)]

    if force_stability:
        for parameter in parameters:
            parameter = np.minimum(parameter, 0.9)
            parameter = np.maximum(parameter, -0.9)


    phi_0 = mu_mean-np.sum(np.array([(parameter @ mu_mean).squeeze() for parameter in parameters]), axis=0, keepdims=True).T

    Sigma = np.zeros([N,N])

    residuals = np.array([(time_series[:,i].reshape([N,1])-phi_0.reshape([N,1])-np.nansum(np.concatenate([(parameters[j]@time_series[:,i-(j+1)*overlap]).reshape([N,1]) for j in range(num_lags)], axis=1),axis=1, keepdims=True).reshape([N,1])).squeeze() for i in range(num_lags*overlap, T)]).T

    for variable_index in range(N):

        valid_start = np.where(~np.isnan(time_series[variable_index]))[0][0]
        valid_residuals = residuals[variable_index, valid_start+ num_lags*overlap:]

        Sigma[variable_index, variable_index] = np.cov(valid_residuals)
    
    eigenvals = np.linalg.eigvals(Sigma)

    if np.max(np.abs(Sigma - Sigma.T))>1e-6:
        raise Warning(f"Sigma is not symmetric:\n {Sigma}")
    
    elif np.min(eigenvals)<0:
        print(f"Sigma is not semipositive definite with min eigenvalue {np.min(eigenvals)}: We will now force it.")

        Sigma = make_positive_semidefinite(Sigma)

    result_dict={
        "Parameters":parameters,
        "Sigma":Sigma,
        "Phi_0":phi_0
    }
    return result_dict



def YW_estimation(data: np.ndarray, maxlags:int, overlap:int=1, sigma_from_OLS:bool=True, force_diagonality_with_YW:bool=False)->dict:
    """
    Perform Yule-Walker estimation for a VAR model.

    Parameters:
    data (np.array): The input data array of shape [num_variables, num_timesteps].
    lags (list): A list of the requested lags.

    Returns:
    list: A list of len(lags) np.arrays, where the i-th array is the coefficient
          matrix of the VAR for the lag given by the i-th element of lags.
    """
    num_variables, num_timesteps = data.shape
    max_lag = maxlags
    
    # Initialize the autocovariance matrices
    autocovariances = [np.zeros((num_variables, num_variables)) for _ in range(max_lag + 1)]

    mu_mean=np.mean(data, axis=1, keepdims=True)
    # mean_squared=mu_mean @ mu_mean.T

    # Compute the autocovariance matrices
    # for overlap_step in range(overlap):
    #     this_overlap_autocovariances = [np.zeros((num_variables, num_variables)) for _ in range(max_lag + 1)]
    #     this_overlap_data=data[:, overlap_step::overlap]
    #     this_overlap_num_timesteps=this_overlap_data.shape[1]
    #     this_overlap_mu_mean=np.mean(this_overlap_data, axis=1, keepdims=True)

    #     for lag in range(max_lag + 1):
    #         # autocovariances[lag]=np.mean(np.array([data[:,t]@data[:,t+lag].T for t in range(num_timesteps-lag)]), axis=0)-mean_squared
    #         for t in range(0, this_overlap_num_timesteps-lag):
    #             this_overlap_autocovariances[lag] += np.outer(this_overlap_data[:, t], this_overlap_data[:, t + lag])
    #         this_overlap_autocovariances[lag] /= (this_overlap_num_timesteps - lag)
    #         this_overlap_autocovariances[lag] -= np.outer(this_overlap_mu_mean, this_overlap_mu_mean)
        
        
    #         autocovariances[lag]+=this_overlap_autocovariances[lag]
    
    # for lag in range(max_lag + 1):
    #     autocovariances[lag]=autocovariances[lag]/overlap

    for current_lag in range(max_lag+1):

        cov_list =  [np.cov(data[:, j: num_timesteps-current_lag*overlap: overlap], data[:, j+current_lag*overlap:: overlap])[:num_variables, num_variables:] for j in range(overlap)]
        autocovariances[current_lag] = np.mean(cov_list, axis=0)

    # Prepare the left-hand side matrix (R)
    Gamma_0 = np.zeros((num_variables * max_lag, num_variables * max_lag))
    for i in range(max_lag):
        for j in range(max_lag):
            if i >= j:
                Gamma_0[i * num_variables:(i + 1) * num_variables, j * num_variables:(j + 1) * num_variables] = autocovariances[i - j]
            else:
                Gamma_0[i * num_variables:(i + 1) * num_variables, j * num_variables:(j + 1) * num_variables] = autocovariances[j - i].T

    # Prepare the right-hand side matrix (P)
    Gamma_1 = np.zeros((num_variables * max_lag, num_variables))
    for i in range(max_lag):
        Gamma_1[i * num_variables:(i + 1) * num_variables, :] = autocovariances[i + 1]

    if force_diagonality_with_YW:
        Gamma_0=np.diag(Gamma_0)
        Gamma_1=np.diag(Gamma_1)

        coefficients=np.diag([Gamma_1[i] / Gamma_0[i] if Gamma_0[i]!=0 else 0 for i in range(len(Gamma_0))])

    # Solve the Yule-Walker equations
    else:
        coefficients = np.linalg.solve(Gamma_0, Gamma_1).T

    # Extract the coefficients for the requested lags
    parameters= [coefficients[:, num_variables*(i-1):num_variables*(i)] for i in range(1, maxlags+1)]

    phi_0 = mu_mean-np.sum(np.array([(parameter @ mu_mean).squeeze() for parameter in parameters]), axis=0, keepdims=True).T

    if sigma_from_OLS:
        residuals = np.array([(data[:,i].reshape([num_variables,1])-phi_0.reshape([num_variables,1])-np.sum(np.concatenate([(parameters[j]@data[:,i-(j+1)*overlap]).reshape([num_variables,1]) for j in range(max_lag)], axis=1),axis=1, keepdims=True).reshape([num_variables,1])).squeeze() for i in range(max_lag*overlap, num_timesteps)])

        Sigma = np.cov(residuals.T)

        
    else:
        Sigma = autocovariances[0]-Gamma_1.T@coefficients.T

    eigenvals = np.linalg.eigvals(Sigma)

    if np.max(np.abs(Sigma - Sigma.T))>1e-6:
        raise Warning(f"Sigma is not symmetric:\n {Sigma}")
    
    elif np.min(eigenvals)<0:
        print(f"Sigma is not semipositive definite with min eigenvalue {np.min(eigenvals)}: We will now force it.")

        Sigma = make_positive_semidefinite(Sigma)

    result_dict={
        "Parameters":parameters,
        "Sigma":Sigma,
        "Phi_0":phi_0
    }
    return result_dict

# def OLS_estimator(data: np.ndarray, lags:list[int])->dict:
#     num_variables, num_timesteps = data.shape
    
#     # Create lagged matrix X and response matrix Y
#     max_lag = max(lags)
#     X = []
#     Y = data[:, max_lag:]

#     for lag in lags:
#         X.append(data[:, max_lag-lag:num_timesteps-lag])
    
#     X = np.concatenate(X, axis=0)
#     X = X.T
    
#     # Add constant term
#     X = np.hstack([np.ones((X.shape[0], 1)), X])
    
#     # Estimate parameters using OLS
#     beta = np.linalg.inv(X.T @ X) @ X.T @ Y.T
    
#     # Extracting Phi_0 (constant term)
#     Phi_0 = beta[0, :].reshape(num_variables, 1)
    
#     # Extracting lag coefficients
#     Parameters = []
#     for i in range(len(lags)):
#         Parameters.append(beta[1 + i*num_variables : 1 + (i+1)*num_variables, :].T)
    
#     # Calculate residuals
#     Y_hat = (X @ beta).T
#     residuals = Y - Y_hat
    
#     # Estimate noise covariance matrix Sigma
#     Sigma = residuals @ residuals.T / (num_timesteps - max_lag - num_variables*len(lags) - 1)
    
#     return {
#         "Parameters": Parameters,
#         "Phi_0": Phi_0,
#         "Sigma": Sigma
#     }

def spectral_decomposition(A: np.ndarray)-> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors= lg.eigh(A)
    indices=np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues=eigenvalues[indices]
    sorted_eigenvectors=eigenvectors[:,indices]

    return np.diag(sorted_eigenvalues), sorted_eigenvectors

def PCA(time_series: np.ndarray)->dict:
    mean=np.mean(time_series, axis=1)
    covariance_matrix=np.cov(time_series)

    L,A= spectral_decomposition(covariance_matrix)

    subtracting_mean=np.tile(mean[:,None], [1, time_series.shape[1]])

    if np.max(np.abs(subtracting_mean))>10**(-7):
        warnings.warn("When doing PCA, the initial time series was not standardized.")

    Y=A.T @ (time_series-subtracting_mean)

    PCA_dict={
        "components": Y,
        "Eigenvalues": np.diag(L),
        "Variables_to_Factors": A.T,
        "Factors_to_Variables": A,
        "Explained_Variance": np.diag(L)/np.sum(np.diag(L))*100,
        "Cumulated_Explained_Variance": [np.sum(np.diag(L)[0:i+1])/np.sum(np.diag(L))*100 for i in range(len(L))]
    }

    return PCA_dict

def handle_input(time_series: np.ndarray, decomposition_intervals:list[int], decomposition_type: str="CF", tails_external_sides=[10,10])->dict:
    final_dict={}

    main_ts= time_series

    components=len(decomposition_intervals)-1
    len_V, len_t=time_series.shape
    
    if decomposition_type=="CF":

        CF_timeseries_input=main_ts.copy()

        decomposition_dict=Christiano_Fitzgerald_filter(CF_timeseries_input, decomposition_intervals, adjust_level=True, generate_external_sides=tails_external_sides)

        final_dict["Max_error"]=np.max(np.abs(time_series-decomposition_dict["Total"]))

        if components==3:
            final_dict["Trend"]=decomposition_dict["component 0"]
            final_dict["Business cicle"]= decomposition_dict["component 1"]
            final_dict["Monthly cicle"]= decomposition_dict["component 2"]

    if decomposition_type=="FT":
        decomposition_dict=Fourier_filter(main_ts, decomposition_intervals[:-1])

        final_dict["Max_error"]=np.max(np.abs(time_series-decomposition_dict["Total"]))

        if components==3:
            final_dict["Trend"]=decomposition_dict[2]
            final_dict["Business cicle"]= decomposition_dict[1]
            final_dict["Monthly cicle"]= decomposition_dict[0]


    final_dict["Decomposition_dict"]=decomposition_dict


    return final_dict

def decompose_df(input_df: pd.DataFrame, decomposition_intervals:list[int], decomposition_type: str="CF", tails_external_sides=[10,10])->dict:
    
    df = input_df.apply(extrapolate_timeseries, axis=1)
    columns = df.columns.to_list()
    dates = columns[2:]
    time_series = df.iloc[:, 2:].to_numpy()
    decomposed_df = {}

    decomposed_dict=handle_input(time_series, decomposition_intervals, decomposition_type=decomposition_type, tails_external_sides=tails_external_sides)
    
    if decomposed_dict["Max_error"]>10**(-8):
        raise Warning("Error too high in decomposition: "+str(decomposed_dict["Max_error"]))
    
    for index, frequency in enumerate(["Trend", "Business cicle", "Monthly cicle"]):
        new_df=pd.DataFrame(columns=dates, data=decomposed_dict[frequency])
        new_df[["Asset", "Datashape"]]=df[["Asset", "Datashape"]].copy()

        new_df["frequency"]=frequency

        new_df=new_df[["Asset", "Datashape", "frequency"]+dates]

        new_df = get_nans_from(df_1=new_df, df_2=input_df)

        decomposed_df[frequency]=new_df


    return {
        "Original time series": time_series,
        "Decomposed df": decomposed_df,
        # "Decomposition dict": decomposed_dict,
        "dates": dates,
        "Assets_dict": df["Asset"].to_dict()
    }

def adjust_base_df(df: pd.DataFrame,  start :str| None = None)->pd.DataFrame:
    df=df.rename(columns={"Time series:": "Asset", "Datashape:": "Datashape", "Variable": "Asset", "DataShape": "Datashape"})

    if start is not None:

        columns = df.columns.to_list()

        my_index = columns.index(start)

        columns_to_drop = columns[my_index+1:]

        df = df.drop(columns=columns_to_drop)

    return df

def PCA_on_decomposed_df(decomposed_dict: dict, max_expl:int=15)->None:

    PCA_dict={}

    expl_var_PCA=pd.DataFrame(columns=["frequency"]+["component "+str(i) for i in range(1,1+max_expl)])

    for index_1, f_and_t in enumerate(decomposed_dict["DR_input"].keys()):

        dates = list(decomposed_dict["DR_input"][f_and_t].columns)[3:]
        
        current_dataframe_of_interest=decomposed_dict["DR_input"][f_and_t].dropna(how="any", axis=0)

        numpy_array=current_dataframe_of_interest[dates].copy().to_numpy().astype(np.float64)

        algo_dict=PCA(numpy_array)

        row_1=[f_and_t]+list(algo_dict["Cumulated_Explained_Variance"][:max_expl])

        expl_var_PCA.loc[len(expl_var_PCA)]=row_1

        PCA_dict[f_and_t]=algo_dict

    PCA_dict["Explained variance"]=expl_var_PCA

    decomposed_dict["PCA"]=PCA_dict

def FSA_on_decomposed_df(decomposed_dict: dict, factors_pool: pd.DataFrame, max_expl:int=15, fixed_order:bool=False)->None:

    FSA_dict={}

    components_df=pd.DataFrame(columns=["frequency and type"]+["component "+str(i) for i in range(1,1+max_expl)])
    expl_var=pd.DataFrame(columns=["frequency and type"]+["component "+str(i) for i in range(1,1+max_expl)])

    for index_1, f_and_t in enumerate(decomposed_dict["DR_input"].keys()):

        dates = list(decomposed_dict["DR_input"][f_and_t].columns)[3:]

        current_dataframe_of_interest=decomposed_dict["DR_input"][f_and_t].dropna(how="any", axis=0)

        current_factors_df=factors_pool[f_and_t]

        if current_factors_df is None:
            raise Warning("factor_df is None")
        
        variables_array=current_dataframe_of_interest[dates].copy().to_numpy().astype(np.float64)

        factors_array=current_factors_df[dates].copy().to_numpy().astype(np.float64)

        algo_dict=algorithm(variables_array, factors_array, max_expl, orthogonal_factors=True, fixed_order = fixed_order)

        factor_list=current_factors_df["Factor"].to_list()
    
        row_1=[f_and_t]+list(algo_dict["Explained Variance"])
        row_2=[f_and_t]+[factor_list[int(i)] for i in algo_dict["Indices"]]
        expl_var.loc[len(expl_var)]=row_1
        components_df.loc[len(components_df)]=row_2

        FSA_dict[f_and_t]=algo_dict

    FSA_dict["components_df"]=components_df
    FSA_dict["Explained variance"]=expl_var

    decomposed_dict["FSA"]=FSA_dict


    
def get_weight_df(location:str, base_df: pd.DataFrame, dates:list)->pd.DataFrame:
    MV_df=pd.read_excel(location, sheet_name="Transformed_data")

    MV_df=MV_df[MV_df["label_4"]=="Value"]
    listah = pd.to_datetime(MV_df.columns[6:], format='%d/%m/%Y')

    # Format datetime columns as desired
    new_list = list(MV_df.columns[:6].to_list())+list(listah.strftime('%m/%Y').str.lstrip('0'))

    MV_df.columns=new_list

    MV_df=MV_df[MV_df["Asset"].isin(base_df["Asset"].to_list())].reset_index(drop=True)

    MV_df=MV_df.dropna(axis=1, how='all')

    weight_df=MV_df.drop(columns=["label_1", "Asset_type", "label_4"])[["Asset"]+dates]

    for asset in base_df["Asset"].to_list():
        if asset not in weight_df["Asset"].to_list():
            weight_df.loc[len(MV_df),:]=[asset]+[np.NaN for i in range(len(weight_df.columns)-1)]

    return weight_df

def my_algorithm(decomposed_dict: dict, size: int, weight_df:pd.DataFrame | None, shifts=list[int]| None):

    my_algorithm_dict={}

    components_df=pd.DataFrame(columns=["frequency and type"]+["component "+str(i) for i in range(1,16)])
    expl_var=pd.DataFrame(columns=["frequency and type"]+["component "+str(i) for i in range(1,16)])

    dates= decomposed_dict["dates"]

    for index_1, f_and_t in enumerate(decomposed_dict["Decomposed df"].keys()):
        current_dataframe_of_interest=decomposed_dict["Decomposed df"][f_and_t]

        current_factors_df=get_factor_df(current_dataframe_of_interest, dates, include=False, weight_df=weight_df, shifts=shifts)

        if current_factors_df is None:
            raise Warning("factor_df is None")
        
        variables_array=current_dataframe_of_interest[dates].copy().to_numpy().astype(np.float64)

        factors_array=current_factors_df[dates].copy().to_numpy().astype(np.float64)

        algo_dict=algorithm(variables_array, factors_array, size, orthogonal_factors=True)

        factor_list=current_factors_df["Factor"].to_list()
    
        row_1=[f_and_t]+list(algo_dict["Explained Variance"])
        row_2=[f_and_t]+[factor_list[int(i)] for i in algo_dict["Indices"]]
        expl_var.loc[len(expl_var)]=row_1
        components_df.loc[len(components_df)]=row_2

        my_algorithm_dict[f_and_t]=algo_dict

    my_algorithm_dict["components_df"]=components_df
    my_algorithm_dict["Explained variance"]=expl_var

    decomposed_dict["MY_ALGO"]=my_algorithm_dict

def beta_to_parameters(beta, dim, number_of_lags):
    # Check if beta has the correct shape
    expected_size = dim * (dim * number_of_lags + 1)
    if beta.shape[0] != expected_size:
        raise ValueError(f"Incorrect shape of beta. Expected {expected_size}, but got {beta.shape[0]}")
    
    # Extract phi_0
    phi_0 = beta[:dim]
    
    # Extract parameters
    parameters = []
    offset = dim
    for i in range(number_of_lags):
        matrix = beta[offset:offset + dim*dim].reshape((dim, dim), order='F')
        parameters.append(matrix)
        offset += dim * dim
    
    return phi_0, parameters

def vectorize_parameters(phi_0, parameters):
    
    dim = phi_0.shape[0]
    number_of_lags = len(parameters)
    
    B = np.concatenate([phi_0.reshape([dim,1])] + parameters, axis=1)

    beta = vectorize_matrix(B)
    
    return beta, dim, number_of_lags

def vectorize_matrix(matrix):
    N,T=matrix.shape

    vector=matrix.reshape([N*T,1], order="F")

    return vector

def data_to_X_1(data: np.ndarray[np.float_], number_of_lags:int=1, jumps:int=1):

    X_2 = data_to_X_2(data, number_of_lags, jumps)

    X_1 = vectorize_matrix(X_2)

    return X_1

def data_to_X_0(data: np.ndarray[np.float_], number_of_lags:int=1, jumps:int=1):

    N,T = data.shape

    X_0=np.ones([T-number_of_lags*jumps, 1])

    for p in range(1,number_of_lags+1):
        X_0=np.concatenate([X_0, np.flip(data[:, number_of_lags*jumps-p*jumps:T-p*jumps], axis=1).T], axis=1)

    return X_0

def data_to_X_2(data: np.ndarray[np.float_], number_of_lags:int=1, jumps:int=1):

    return np.flip(data[:, number_of_lags*jumps:], axis=1)

def OLS_new_estimator(data: np.ndarray[np.float_], number_of_lags:int=1, jumps:int=1):

    N,T = data.shape
    
    X_1 = data_to_X_1(data=data, number_of_lags=number_of_lags, jumps=jumps)
    X_0 = data_to_X_0(data=data, number_of_lags=number_of_lags, jumps=jumps)

    I_N = np.identity(N)

    beta_OLS = np.kron(np.linalg.inv(X_0.T @ X_0) @ X_0.T, I_N) @ X_1

    beta_OLS = np.where(np.abs(beta_OLS)<10**(-7), 0, beta_OLS)

    phi_0, parameters = beta_to_parameters(beta_OLS, dim=N, number_of_lags=number_of_lags)

    U = X_1 - kron_identity(N, X_0) @ beta_OLS

    V=U.reshape([N, T-number_of_lags*jumps], order="F")

    Sigma= np.cov(V)


    return {
        "Parameters": parameters,
        "beta_OLS": beta_OLS, 
        "Phi_0": phi_0,
        "Sigma": Sigma
    }

def Sigma_to_Lambda(Sigma: np.ndarray[np.float_], data: np.ndarray[np.float_], number_of_lags:int, jumps: int):

    N,T = data.shape

    Xi_inv = np.linalg.inv(identity_kron(T-number_of_lags*jumps, Sigma))

    X_0 = data_to_X_0(data=data, number_of_lags=number_of_lags, jumps=jumps)

    X_0_kron_I_N = kron_identity(N, X_0)

    Lambda = np.linalg.inv(X_0_kron_I_N.T @ Xi_inv @ X_0_kron_I_N)

    return Lambda

def get_S_L(data: np.ndarray[np.float_], number_of_lags:int=1, jumps:int=1):

    X_0 = data_to_X_0(data, number_of_lags, jumps)

    X_2 = data_to_X_2(data, number_of_lags, jumps)

    pseudo_inv = np.linalg.inv(X_0.T @ X_0)

    B_tilde = X_2 @ X_0 @ pseudo_inv

    part = X_2 - B_tilde @ X_0.T
    
    S_L = part @ part.T

    return S_L

def get_S_L_from_beta(beta, X_0, X_2):

    B_tilde = np.reshape(beta, [X_2.shape[0], X_0.shape[1]], order="F")

    part = X_2 - B_tilde @ X_0.T
    
    S_L = part @ part.T

    return S_L


def kron_identity(K, matrix : np.ndarray[np.float_]):

    N,T=matrix.shape

    new_matrix=np.zeros([K*N, K*T])

    for i in range(N):
        for j in range(T):
            new_matrix[i*K:(i+1)*K, j*K:(j+1)*K] = np.identity(K)*matrix[i,j]

    return new_matrix

def identity_kron(K, matrix : np.ndarray[np.float_]):

    N,T=matrix.shape

    new_matrix=np.zeros([K*N, K*T])

    for i in range(K):
        new_matrix[i*N:(i+1)*N, i*T:(i+1)*T] = matrix

    return new_matrix

def multilinear_multivariate_regression(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Perform multilinear multivariate regression where each of the N response variables 
    is regressed on the M factors. The factors are available for all T observations, 
    but some of the response variables have missing observations at the beginning.

    Parameters:
    X (np.ndarray): Factors array of shape (M, T)
    Y (np.ndarray): Response variables array of shape (N, T) with potential leading np.nan values

    Returns:
    np.ndarray: Coefficients matrix of shape (N, M)
    """
    
    N, T = Y.shape
    M = X.shape[0]
    coefficients = np.zeros((N, M))

    for i in range(N):
        # Find the first non-NaN index for the current time series
        first_valid_index = np.where(~np.isnan(Y[i].astype(float)))[0][0]
        
        # Slice X and Y based on the first valid index
        X_trimmed = X[:, first_valid_index:].T  # Shape: (T', M)
        Y_trimmed = Y[i, first_valid_index:]    # Shape: (T',)
        
        # Perform linear regression
        model = LinearRegression()
        model.fit(X_trimmed, Y_trimmed)
        
        # Store the coefficients
        coefficients[i, :] = model.coef_

    return coefficients


def make_positive_semidefinite(matrix):
    # Ensure the input is a numpy array
    matrix = np.array(matrix)
    
    # Step 1: Diagonalize the matrix (A = Q L Q^T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Step 2: Replace negative eigenvalues with zero
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Step 3: Reconstruct the matrix (Q new_L Q^T)
    positive_semidefinite_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    return positive_semidefinite_matrix

def strongly_diagonalize(matrix):

    det_full = np.linalg.det(matrix)

    diagonal = np.diag(np.diag(matrix))

    det_diag = np.linalg.det(diagonal)

    return diagonal * det_full / det_diag

def var_stability_test(B):
    """
    Given a VAR coefficient matrix B of shape (K, pK+1), where K is the dimensionality 
    of the vector and p is the number of lags, this function checks the stability condition 
    by computing the companion matrix and returns the largest eigenvalue (in terms of magnitude).
    
    Parameters:
    B (numpy.ndarray): Coefficient matrix of shape (K, pK+1).
    
    Returns:
    float: The largest eigenvalue (in terms of magnitude).
    """
    K = B.shape[0]
    p = (B.shape[1] - 1) // K  # Number of lags

    # Construct the companion matrix
    companion_matrix = np.zeros((K * p, K * p))
    
    # Top block: Coefficients B[:, 1:] (excluding the intercept term)
    companion_matrix[:K, :] = B[:, 1:]
    
    # Lower blocks: Identity matrices for shifting
    if p > 1:
        companion_matrix[K:, :-K] = np.eye(K * (p - 1))
    
    # Compute eigenvalues of the companion matrix
    eigenvalues = np.linalg.eigvals(companion_matrix)
    
    # Return the largest eigenvalue in terms of magnitude
    return np.max(np.abs(eigenvalues))


def mardia_test(X: np.ndarray, Sigma: np.ndarray =None, zero_mean: bool= False):
    """
    Perform Mardia's test for multivariate normality.
    
    Parameters:
    X (np.ndarray): A numpy array of shape (N, T), where N is the dimensionality of the 
                    multivariate data, and T is the number of observations.
    
    Sigma (np.ndarray): If we want to test it directly on Sigma.
    zero_mean (bool): If we suppose the error to have zero mean.
    
    Returns:
    A (float): Multivariate skewness statistic.
    B (float): Multivariate kurtosis statistic.
    p_value_A (float): p-value for multivariate skewness.
    p_value_B (float): p-value for multivariate kurtosis.
    """
    # Ensure input is of the correct shape
    N, T = X.shape
    
    # Calculate the mean of the data
    if zero_mean:
        mean_X = np.zeros([N,1])
    else:
        mean_X = np.mean(X, axis=1, keepdims=True)
    
    # Center the data (X - mean_X)
    X_centered = X - mean_X
    
    # Calculate the sample covariance matrix
    if Sigma is None:
        S = np.cov(X_centered, bias=True)  # bias=True ensures normalization by T instead of (T-1)
    else:
        S = Sigma
    
    # Inverse of the covariance matrix
    S_inv = np.linalg.inv(S)
        
    # Multivariate skewness (A)
    A =  np.sum((X_centered.T @ S_inv @ X_centered)**3) / (6*T)
    
    # Multivariate kurtosis (B)
    B = np.sqrt(T/(8*N*(N+2))) * (1 / T * np.sum(np.diag((X_centered.T @ S_inv @ X_centered)**2)) - N * (N + 2))
    
    # Skewness p-value (chi-square distribution with N(N+1)(N+2)/6 degrees of freedom)
    df_A = N * (N + 1) * (N + 2) / 6

    if N<20:
        c = (T+1)*(T+3)*(N+1) / (T*(T+1)*(N+1)-6)
        A = A*c
    p_value_A = 1 - chi2.cdf(A, df = df_A)
    
    # Kurtosis p-value (normal distribution)
    p_value_B = 2 * (1 - norm.cdf(np.abs(B)))  # Two-tailed test
    
    return A, B, p_value_A, p_value_B, df_A


def multivariate_portmanteau_test(V, h):
    """
    Perform the Multivariate Portmanteau Test (Hosking's M-Statistic) for autocorrelation in residuals.
    
    Parameters:
    V : np.ndarray
        Residual matrix of shape (N, T), where N is the number of variables and T is the number of observations.
    h : int
        The maximum lag up to which to test for autocorrelation.
    
    Returns:
    M_stat : float
        The test statistic for the Portmanteau test.
    p_value : float
        The p-value associated with the test statistic.
    """
    N, T = V.shape
    
    # Subtract the mean from each variable (center the data)
    V_centered = V - V.mean(axis=1, keepdims=True)
    
    # Compute the sample covariance matrix of centered residuals
    S = np.cov(V_centered, bias=True)
    
    # Initialize the test statistic
    M_stat = 0.0
    
    # Compute the autocovariance matrices for each lag up to h
    for lag in range(1, h + 1):
        # Autocovariance matrix at lag 'lag'
        A_k = np.zeros((N, N))
        for t in range(lag, T):
            A_k += np.outer(V_centered[:, t], V_centered[:, t - lag])
        A_k /= (T - lag)
        
        # Update the test statistic
        M_stat += np.trace(np.linalg.inv(S) @ A_k @ np.linalg.inv(S) @ A_k.T)
    
    # Multiply by T (not T^2)
    M_stat *= T
    
    # Degrees of freedom for the chi-square distribution
    df = N**2 * h
    
    # Compute the p-value
    p_value = 1 - chi2.cdf(M_stat, df)
    
    return M_stat, p_value


def multivariate_lm_test(V, h):
    """
    Perform the Multivariate Lagrange Multiplier (LM) Test for autocorrelation in residuals.
    
    Parameters:
    V : np.ndarray
        Residual matrix of shape (N, T), where N is the number of variables and T is the number of observations.
    h : int
        The specific lag at which to test for autocorrelation.
    
    Returns:
    LM_stat : float
        The test statistic for the LM test.
    p_value : float
        The p-value associated with the test statistic.
    """
    N, T = V.shape
    
    # Subtract the mean from each variable (center the data)
    V_centered = V - V.mean(axis=1, keepdims=True)
    
    # Step 1: Compute the sample covariance matrix of centered residuals
    S = np.cov(V_centered, bias=True)
    
    # Step 2: Compute the residual autocovariance matrix at lag h
    A_h = np.zeros((N, N))
    for t in range(h, T):
        A_h += np.outer(V_centered[:, t], V_centered[:, t - h])
    A_h /= (T - h)
    
    # Step 3: Compute the test statistic
    LM_stat = T * np.trace(A_h.T @ np.linalg.inv(S) @ A_h)
    
    # Degrees of freedom for the chi-square distribution
    df = N**2
    
    # Compute the p-value
    p_value = 1 - chi2.cdf(LM_stat, df)
    
    return LM_stat, p_value


def vectorized_rowwise_mean_cov(arr):
    # Step 1: Compute row-wise mean, ignoring NaNs
    row_means = np.nanmean(arr, axis=1)
    
    # Step 2: Compute the covariance matrix in a vectorized manner
    
    # We multiply every pair of rows with each other (without looping)
    # Use np.newaxis to add an axis so broadcasting works properly
    elementwise_product = np.nanmean(arr[:, np.newaxis, :] * arr[np.newaxis, :, :], axis=2)
    
    # Compute covariance using the formula: cov(X, Y) = E[XY] - E[X]E[Y]
    covariance_matrix = elementwise_product - np.outer(row_means, row_means)

    print(np.min(np.linalg.eigvals(covariance_matrix)))
    
    return row_means, covariance_matrix