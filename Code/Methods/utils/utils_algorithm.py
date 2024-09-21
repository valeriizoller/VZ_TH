import numpy as np
import scipy as sp
from typing import Callable, List, Tuple
import warnings

def project_vec(v: np.ndarray, u: np.ndarray, tol: float = 10**(-9)) -> np.ndarray:
    """
    Projects the vector v onto u.

    Args:
        v: np.array
            The vector to be projected.
        u: np.array
            The vector on which to project.
        tol: float
            Tolerance defining when to consider a vector to be zero. Default is 10^(-4).

    Returns:
        np.array: The projected vector.
    """
    num = v @ u
    den = u @ u

    if den > tol:
        coeff = num / den
    elif num < tol:
        coeff = 0
    else:
        raise Warning("Division by zero in projection")

    return coeff * u

def project_coeff(v: np.ndarray, u: np.ndarray, tol: float = 10**(-9)) -> float:
    """
    Projects the vector v onto u.

    Args:
        v: np.array
            The vector to be projected.
        u: np.array
            The vector on which to project.
        tol: float
            Tolerance defining when to consider a vector to be zero. Default is 10^(-4).

    Returns:
        np.array: The projected vector.
    """
    num = v @ u
    den = u @ u

    if den > tol:
        coeff = float(num / den)
    elif np.abs(num) < 10**7*tol:
        coeff = 0
    else:

        raise Warning("Division by zero in projection: v@u="+str(num)+" \n u@u="+str(den))

    return coeff

def orthogonalize(vector: np.ndarray, eq_constraints: np.ndarray) -> np.ndarray:
    """
    Projects a vector onto the subspaces defined by eq_constraints.

    Args:
        vector: np.array
            Vector to project.
        eq_constraints: np.array
            Constraints on which to project.

    Returns:
        np.array: The projected vector.
    """
    new_vec_constraints = gram_schmidt(eq_constraints)
    projection = np.sum(np.array([project_vec(vector, u) for u in new_vec_constraints]), axis=0)
    return vector - projection

def normalize(vector: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector.

    Args:
        vector: np.array
            Vector to normalize.

    Returns:
        np.array: Normalized vector.
    """
    den = np.linalg.norm(vector)
    if den == 0:
        return vector
    else:
        return vector / den
    
def invert_dictionary(original_dict):

    return {v: k for k, v in original_dict.items()}

def gram_schmidt(matrix: np.ndarray) -> np.ndarray:
    """
    Performs Gram-Schmidt orthogonalization on the given matrix.

    Args:
        matrix: np.array
            Matrix on which to perform Gram-Schmidt orthogonalization.

    Returns:
        np.array: The matrix after Gram-Schmidt orthogonalization.
    """
    new_matrix = np.copy(matrix)

    for i, vec in enumerate(matrix):
        new_matrix[i] = vec - np.sum(np.array([project_vec(vec, u) for u in new_matrix[:i]]), axis=0)

    return new_matrix

def get_correlation_matrix( time_series : np.ndarray , ddof : int = 1 )-> np.ndarray:

    covariance_matrix   =   np.cov(time_series, ddof=ddof)

    stds                =   np.diag(np.sqrt(np.diag(covariance_matrix)))
    stds_1              =   np.where(stds<10**(-8), 0, np.power(stds.astype(np.float64),-1))

    corr                =   stds_1@ covariance_matrix @ stds_1

    return corr

def standardize(time_series : np.ndarray , ddof : int = 1)-> np.ndarray:

    mean=np.mean(time_series, axis=1, keepdims=True)

    std=np.std(time_series, axis=1, keepdims=True, ddof=ddof)

    standardized_time_series=(time_series-mean)/std

    return standardized_time_series

def regressors_coeff(regressors: np.ndarray) -> np.ndarray:
    second_moment = regressors @ regressors.T
    var_f_i = np.diag(np.cov(regressors))
    second_moment_diag = np.diag(second_moment)**2

    return np.divide(var_f_i, second_moment_diag, out=np.zeros_like(var_f_i), where= (second_moment_diag!=0)) #np.where(second_moment_diag==0, 0 , var_f_i/second_moment_diag) 

def traces_cov_decomposition(O_i: np.ndarray, regressors : np.ndarray)-> np.ndarray:
    matrix= O_i @ regressors.T
    squared_matrix= matrix**2
    sums= np.sum(squared_matrix, axis=0)
    
    return sums

def algorithm_phase(variables: np.ndarray, regressors: np.ndarray, coeff :np.ndarray|None = None, orthogonal_factors: bool = True, fixed_order:bool=False, phase: int|None =None)-> dict:

    return_dict={}

    N=variables.shape[0]
    M=regressors.shape[0]
    n=variables.shape[1]

    flattened_variables=np.where(np.abs(variables)<10**(-8), 0, variables)

    

    if orthogonal_factors: 
        coeff=regressors_coeff(regressors)
        flattened_regressors=np.where(np.abs(regressors)<10**(-8), 0, regressors)
    else:
        flattened_regressors=regressors


    sums = traces_cov_decomposition(flattened_variables, flattened_regressors)
    traces = coeff * sums
    trace = np.trace(np.cov(flattened_variables))
    
    if fixed_order:
        if phase is None:
            raise ValueError("Phase has to be an integer!")
        if phase>M-1:
            warnings.warn(f"Capping phase at {M}.")
            phase = M-1

        choosen_index=phase
    else:
        choosen_index=np.argmax(traces)

    choosen_trace=traces[choosen_index]

    f_i=flattened_regressors[choosen_index,:].reshape([n,1])
    a_i=np.array([project_coeff(v.squeeze(), f_i.squeeze()) for v in flattened_variables]).reshape([N,1])

    if orthogonal_factors:
        b_i=np.array([project_coeff(v.squeeze(), f_i.squeeze()) for v in flattened_regressors]).reshape([M,1])

    if np.abs(choosen_trace-np.trace(np.cov(a_i@f_i.T)))>10**(-4):
        raise Warning("Trace mismatch:\n Analitical =   "+str(choosen_trace)+"\n Effective:   "+str(np.trace(np.cov(a_i@f_i.T))))

    residual=flattened_variables-a_i@f_i.T

    if (np.abs(np.trace(np.cov(residual))-(trace-choosen_trace))>10**(-8)) and (np.abs(np.trace(np.cov(residual))-(trace-choosen_trace))/(trace-choosen_trace)>0.01):
        warnings.warn("Orthogonality is falting:\n Difference =   "+str(np.abs(np.trace(np.cov(residual))-(trace-choosen_trace)))
                      +"\n Residual:   "+str(np.trace(np.cov(residual)))
                      +"\n Block:   "+str(trace-choosen_trace)
                      )

    flattened_residual=np.where(np.abs(residual)<10**(-8), 0, residual)

    if orthogonal_factors:
        residual_factors=flattened_regressors-b_i@f_i.T
        flattened_residual_factors=np.where(np.abs(residual_factors)<10**(-8), 0, residual_factors)
        return_dict["Residual Factors"]= flattened_residual_factors

    return_dict["Residual"]=flattened_residual
    return_dict["Choosen index"]=choosen_index
    return_dict["Factor"]=f_i
    return_dict["Correlation coefficients"]=a_i
    return_dict["Full trace"]=trace
    return_dict["Vector of traces"]=traces
    return_dict["Choosen trace"]= choosen_trace
    
    return return_dict

def algorithm(variables: np.ndarray, regressors: np.ndarray, phases: int, orthogonal_factors: bool = True, fixed_order:bool=False)-> dict:

    return_dict={}

    N=variables.shape[0]
    n=variables.shape[1]

    factors=np.zeros([phases, n])

    eigenvectors=np.zeros([N, phases])

    traces=np.zeros(phases)

    cumulated_traces=np.zeros(phases)

    total_trace=np.trace(np.cov(variables))

    indeces=np.zeros(phases)

    residual=variables

    if orthogonal_factors: coeff = None
    else: coeff=regressors_coeff(regressors)

    for phase in range(phases):

        # print(f"phase {phase+1} of {phases}")

        new_dict=algorithm_phase(residual, regressors, coeff, orthogonal_factors, fixed_order=fixed_order, phase=phase)

        residual=new_dict["Residual"]

        factors[phase,:]=new_dict["Factor"].T.squeeze()

        eigenvectors[:, phase]=new_dict["Correlation coefficients"].squeeze()

        traces[phase]=new_dict["Choosen trace"]

        cumulated_traces[phase]=np.sum(traces)

        indeces[phase]=new_dict["Choosen index"]

        if orthogonal_factors: regressors=new_dict["Residual Factors"]


    return_dict["Factors"]= factors
    return_dict["Factors_to_Variables"]= eigenvectors
    return_dict["Traces"]= traces
    return_dict["Cumulated traces"]= cumulated_traces
    return_dict["Explained Variance"]= cumulated_traces/total_trace
    return_dict["Indices"]=indeces
    return_dict["Residual"]=residual

    return return_dict















