import numpy as np
from numpy import linalg as lg
from tqdm import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline


def discrete_fourier_transform(time_series_matrix):
    dates = time_series_matrix.shape[1]
    frequencies = 2 * np.pi * np.arange(dates) / dates

    spectral_arg= np.arange(dates).reshape([dates,1])@frequencies.reshape([1,dates]) 
    cos_coeffs = np.cos(spectral_arg)
    sin_coeffs = np.sin(spectral_arg)

    A = np.dot(time_series_matrix, cos_coeffs)/dates
    B = np.dot(time_series_matrix, sin_coeffs)/dates

    return {"Fourier frequencies": frequencies, "A": A, "B": B, "Periods":  2 * np.pi/frequencies, "time_line":np.arange(dates)}

def Fourier_filter(time_series_matrix, barriers):
    Fourier_dict=discrete_fourier_transform(time_series_matrix)

    complete_barriers=barriers

    times=Fourier_dict["time_line"]

    filter_dict={}

    filter_dict["full"]=time_series_matrix

    filter_dict["Total"]=np.zeros(time_series_matrix.shape)

    for index, barrier in enumerate(complete_barriers):
        sliced_indices=[]

        if index != (len(barriers[:-1])):
            initial_barrier=barrier
            final_barrier=complete_barriers[index+1]

            for i, num in enumerate(Fourier_dict["Periods"]):
                if initial_barrier <= num < final_barrier:
                    sliced_indices.append(i)

        else:
            initial_barrier=barrier
            for i, num in enumerate(Fourier_dict["Periods"]):
                if initial_barrier <= num:
                    sliced_indices.append(i)

        len_V=time_series_matrix.shape[0]
        len_f=len(sliced_indices)
        len_t=len(times)
        
        sliced_freq=Fourier_dict["Fourier frequencies"][sliced_indices]
        sliced_A=Fourier_dict["A"][:,sliced_indices]
        sliced_B=Fourier_dict["B"][:,sliced_indices]

        arg_matrix=np.array(sliced_freq).reshape([len_f,1])@times.reshape([1,len_t])

        new_A=np.tile(sliced_A.reshape([len_V, len_f, 1]), [1,1,len_t])
        new_B=np.tile(sliced_B.reshape([len_V, len_f, 1]), [1,1,len_t])

        cos_matrix=np.tile(np.cos(arg_matrix).reshape([1,len_f, len_t]), [len_V,1,1])
        sin_matrix=np.tile(np.sin(arg_matrix).reshape([1,len_f, len_t]), [len_V,1,1])

        sliced_timeseries=np.sum(new_A*cos_matrix+new_B*sin_matrix, axis=1)

        filter_dict[index]=sliced_timeseries

        filter_dict["Total"]+=sliced_timeseries

    return filter_dict


def trigcardinal(x, N):
    # Precompute constants
    Npi = N * np.pi
    half_Npi = Npi / 2
    half_pi = np.pi / 2
    
    # Compute common values
    sin_half_pi_x = np.sin(half_pi * x)
    near_zero = np.abs(x) < 1e-6
    
    # Initialize tau
    tau = np.ones_like(x)

    if N % 2 == 1:  # ODD
        # For odd N, use sine
        sin_N_half_pi_x = np.sin(half_Npi * x)
        
        # Avoid recalculating values for non-zero sin_half_pi_x
        non_zero_mask = sin_half_pi_x != 0
        tau[non_zero_mask] = sin_N_half_pi_x[non_zero_mask] / (N * sin_half_pi_x[non_zero_mask])
    else:  # EVEN
        # For even N, use tangent
        sin_N_half_pi_x = np.sin(half_Npi * x)
        tan_half_pi_x = np.tan(half_pi * x)
        
        # Avoid recalculating values for non-zero sin_half_pi_x
        non_zero_mask = sin_half_pi_x != 0
        tau[non_zero_mask] = sin_N_half_pi_x[non_zero_mask] / (N * tan_half_pi_x[non_zero_mask])
    
    # Handle small x values to avoid division issues
    tau[near_zero] = 1

    return tau

def discrete_fourier_transform_2(time_series_matrix, overlay_number):
    number_of_paths, N, T = time_series_matrix.shape
    new_T = (T - 1) * overlay_number + 1

    original_timeline = np.array([2*m/T for m in range(T)])
    new_timeline = np.linspace(original_timeline[0], original_timeline[-1], new_T)

    # Precompute the new time matrix and reshaped time series matrix
    new_time_matrix = new_timeline.reshape(1, 1, new_T)
    time_series_reshaped = time_series_matrix.reshape(number_of_paths, N, T, 1)

    # Initialize the interpolated time series
    interpolated_ts = np.zeros((number_of_paths, N, new_T))

    for k in range(T):
        # Compute differences once per iteration
        time_diff = new_time_matrix - original_timeline[k]

        # Calculate the trigcardinal values with caching
        cardinal_values = trigcardinal(time_diff, T)

        # Accumulate the interpolated results
        interpolated_ts += time_series_reshaped[:, :, k, :] * cardinal_values

    return interpolated_ts



def generate_wanted_matrix(values):
    size = len(values)
    reverse_values=values[1:]
    reverse_values.reverse()
    new_values=reverse_values+values
    matrix = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        row= new_values[size-1-i:2*size-i-1]
        matrix[i,:] = np.array(row)
    return matrix
        
def Christiano_Fitzgerald_filter(data, barriers, adjust_level=False, generate_external_sides=[0,0]):

    time_series_matrix=data.copy()

    if max(generate_external_sides)>0:
        first_column=time_series_matrix[:,:1]
        last_column=time_series_matrix[:,-1:]
        time_series_matrix=np.concatenate([np.tile(first_column, [1, generate_external_sides[0]]), time_series_matrix,np.tile(last_column, [1,generate_external_sides[1]])], axis=1)

    variables, dates               =   time_series_matrix.shape
    original_dates= data.shape[1]
    times               =   np.arange(1, dates+1)

    frequencies         =   [2*np.pi/barrier for barrier in barriers]
    frequencies.sort()

    len_t               =   len(times)

    filter_dict={}
    filter_dict["frequencies"]=frequencies

    for index, w_1 in enumerate(frequencies[:-1]):
        w_2 = frequencies[index+1]

        g_0=(w_2-w_1)/np.pi

        g_l=[(np.sin(w_2*l)-np.sin(w_1*l))/(np.pi*l) for l in np.arange(1, len_t)]

        g_l=[g_0]+g_l

        g_last=[(-0.5*g_0-np.sum(g_l[-i:])) for i in range(1,len_t+1)]

        provisional_matrix=generate_wanted_matrix(g_l)

        provisional_matrix[-1,:]=g_last

        g_first=np.sum(provisional_matrix[1:,:],axis=0)#[(-0.5*g_0-np.sum(g_l[1:])) for i in range(1,len_t+1)]

        provisional_matrix[0,:]=g_first

        y = time_series_matrix @ provisional_matrix

        if max(generate_external_sides)>0:
            y=y[:, generate_external_sides[0]:-generate_external_sides[1]]

        if adjust_level==True:

            if index==0:
                total_level=np.zeros(y.shape)

            total_level+=y

            if w_1==frequencies[-2]:
                diff=time_series_matrix[:,generate_external_sides[0]+5]-total_level[:,5]

                diff=np.tile(diff.reshape([variables,1]), [1,original_dates])

                total_level+=diff

                filter_dict["component 0"]+=diff
                
        filter_dict["component "+str(index)]=y

        if adjust_level: filter_dict["Total"]=total_level

        filter_dict["provisional_matrix "+str(index)]=provisional_matrix

    return filter_dict

def extrapolate_flat_timeseries(row):

    # Extract the time series data (ignoring the first two columns)
    timeseries = row.to_numpy()[2:]
    
    # Find the index of the first non-NaN value
    first_valid_index = np.where(~np.isnan(timeseries.astype(float)))[0][0]

    if first_valid_index!=0:
               
        # Fill in the NaNs with the extrapolated data
        timeseries[0:first_valid_index] = timeseries[first_valid_index]
        
        # Update the original row with the filled data
        row[2:] = timeseries
    
    return row

def extrapolate_timeseries(row):
    # Extract the time series data (ignoring the first two columns)
    datashape = row.to_numpy()[1]

    timeseries = row.to_numpy()[2:]
    
    # Find the index of the first non-NaN value
    first_valid_index = np.where(~np.isnan(timeseries.astype(float)))[0][0]

    if first_valid_index!=0:
        # Extract the available non-NaN data from the timeseries
        available_data = timeseries[first_valid_index:]
        
        # Calculate how many NaNs need to be filled
        num_nans_to_fill = first_valid_index
        
        # Repeat the available data backward to fill NaNs
        repeated_data = available_data[-num_nans_to_fill:]
        
        # Find the shift to ensure continuity
        shift_value = available_data[0] - repeated_data[-1]

        if datashape == "Level":
            shifted_data = repeated_data + np.linspace(-shift_value, shift_value, len(repeated_data))
        else:
            shifted_data = repeated_data + shift_value
        
        # Fill in the NaNs with the extrapolated data
        timeseries[first_valid_index-len(shifted_data):first_valid_index] = shifted_data
        
        # Update the original row with the filled data
        row[2:] = timeseries

        row = extrapolate_timeseries(row)
    
    return row

def levels_to_loglevels(row, shift=0.011):

    datashape = row.to_numpy()[1]

    if datashape=="Level":

        timeseries = row.to_numpy()[2:]

        asset_name = row.to_numpy()[0]

        check_1 = ("RV" in asset_name)
        check_2 = (asset_name[:4]=="EQPE")
        check_3 = (asset_name[:4]=="UNEM")
        check_4 = (asset_name[:4]=="EQDY")
        check_5 = (asset_name[:4]=="TBSR")
        check_6 = (asset_name[:4]=="NGLR")

        if check_1 or check_2 or check_3 or check_4 or check_5 or check_6:

            timeseries = np.log(timeseries.astype(float)+shift)

            row.iloc[2:] = timeseries
            row.iloc[1] = "LogLevel"

        
    return row

def loglevels_to_levels(row, shift=0.011):

    datashape = row.to_numpy()[1]

    if datashape=="LogLevel":

        timeseries = row.to_numpy()[2:]

        timeseries = np.exp(timeseries.astype(float))-shift

        row.iloc[2:] = timeseries
        row.iloc[1] = "Level"

    return row

def forecast_loglevels_to_levels(array, shift=0.011):
    
    new_array = np.exp(array)-shift

    return new_array

def get_nans_from(df_1, df_2):

    asset_names = df_1["Asset"].to_list()

    for index, asset_name in enumerate(asset_names):

        row_2 = df_2.iloc[index,:]

        timeseries_2 = row_2.to_numpy()[3:]
        
        # Find the index of the first non-NaN value
        first_valid_index = np.where(~np.isnan(timeseries_2.astype(float)))[0][0]

        df_1.iloc[index, 3:first_valid_index]=np.nan

    return df_1


def keep_first_n_non_nan(arr, n):
    result = np.zeros_like(arr)
    
    for i in range(arr.shape[0]):
        non_nan_indices = np.where(~np.isnan(arr[i]))[0]
        if len(non_nan_indices) > n:
            first_n_indices = non_nan_indices[:n]
        else:
            first_n_indices = non_nan_indices
        
        result[i, first_n_indices] = arr[i, first_n_indices]
    
    return result    

def multisplines(t, f, t_interp):
    # Initialize the output array for interpolated values
    f_interp = np.zeros((*f.shape[:-1], len(t_interp)))

    # Apply interpolation across the last axis (time series)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            spline = InterpolatedUnivariateSpline(t, f[i, j, :], k=3, ext=0)
            f_interp[i, j, :] = spline(t_interp)
    
    return f_interp

