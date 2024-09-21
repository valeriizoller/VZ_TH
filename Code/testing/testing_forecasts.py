import sys
import os
sys.path.append(os.path.abspath(r"C:\Users\valeriiz\OneDrive - ORTEC Finance\Desktop\Thesis\Code"))
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import xlsxwriter
import pandas as pd
from scipy.stats import norm, gmean
from tqdm import tqdm
from Methods.utils.utils_methods import adjust_base_df

number_of_paths=2000
bootstrap_number = 10

starting_date_sim = "1/2015"
end_date_test = "12/2016"

factors_from = ["FSA" , "PCA", "ASF"]
core_models =["VAR" , "BVAR", "RW"]
prior_name = ["Minnesota_prior" , "Jeffreys_prior" , "Zellners_prior" , "Conj_Norm_Inv_Wish_Prior" , "Ind_Norm_Inv_Wish_Prior" , "uniform_prior"]
directory = "C:\\Users\\valeriiz\\OneDrive - ORTEC Finance\\Desktop\\Thesis\\Results\\"
benchmark_name = "VAR_PCA"

testing_df = adjust_base_df(pd.read_excel(r"C:\Users\valeriiz\OneDrive - ORTEC Finance\Desktop\Thesis\Data\Thesis data set - extended data.xlsx", sheet_name="Final Data"))
testing_assets_names_all = testing_df["Asset"].to_list()
testing_assets_names_elite = pd.read_excel(r"C:\Users\valeriiz\OneDrive - ORTEC Finance\Desktop\Thesis\Data\Thesis data set.xlsx", sheet_name="Outcome evaluation")["Time series:"].to_list()


def single_ts_rmse(actual_ts: np.ndarray, estimated_ts: np.ndarray):

    assert actual_ts.shape==estimated_ts.shape

    return np.sqrt(np.sum((actual_ts-estimated_ts)**2))

def mean_path_rmse(actual_ts:np.ndarray, ts_array: np.ndarray):

    assert ts_array.shape[1]==len(actual_ts)

    return np.mean(np.sqrt(np.mean((ts_array-actual_ts)**2, axis=1)))

def point_rmse(actual_ts:np.ndarray, ts_array: np.ndarray):

    assert ts_array.shape[1]==len(actual_ts)

    return np.sqrt(np.mean((ts_array-actual_ts)**2, axis=0))

def single_ts_mae(actual_ts: np.ndarray, estimated_ts: np.ndarray):

    assert actual_ts.shape==estimated_ts.shape

    return np.mean(np.abs(actual_ts-estimated_ts))

def mean_path_mae(actual_ts:np.ndarray, ts_array: np.ndarray):

    assert ts_array.shape[1]==len(actual_ts)

    return np.mean(np.mean(np.abs(ts_array-actual_ts), axis=1))

def point_mae(actual_ts:np.ndarray, ts_array: np.ndarray):

    assert ts_array.shape[1]==len(actual_ts)

    return np.mean(np.abs(ts_array-actual_ts), axis=0)

def single_ts_correlation(actual_ts: np.ndarray, estimated_ts: np.ndarray):

    assert actual_ts.shape==estimated_ts.shape
    
    cov_matrix = np.cov(actual_ts, estimated_ts)

    assert cov_matrix.shape == (2,2)

    return cov_matrix[0,1]/np.sqrt(cov_matrix[0,0]*cov_matrix[1,1])

def mean_path_correlation(actual_ts:np.ndarray, ts_array: np.ndarray):

    assert ts_array.shape[1]==len(actual_ts)

    N, T = ts_array.shape

    normalized_ts_array = (ts_array-np.mean(ts_array))/np.std(ts_array)
    normalized_actual_ts= (actual_ts-np.mean(actual_ts))/np.std(actual_ts)

    return np.mean(np.dot(normalized_ts_array, normalized_actual_ts))/T

def energy_score_per_asset(actual_ndarray, scenarios):

    first_term  = np.mean(np.sum((scenarios - actual_ndarray)**2, axis=1))

    mean = np.mean(scenarios, axis=0)

    diag = np.diag(scenarios @ scenarios.T).squeeze()

    second_term = np.mean(diag) - mean @ mean

    return first_term - second_term

def variogram_score_per_asset(actual_ndarray, scenarios, p: float):

    n = len(actual_ndarray)

    total_sum = 0

    for i in range(n):
        for j in range(n):

            if i!=j:

                first_diff = np.abs(actual_ndarray[i]-actual_ndarray[j])**p
                second_diff= np.mean(np.abs(scenarios[:, i]-scenarios[:, j])**p)

                total_sum += ((first_diff-second_diff)**2)/(np.abs(i-j))


    return total_sum

def old_point_crps(actual_ndarray, scenarios):

    N = len(actual_ndarray)
    M = scenarios.shape[0]

    sorted_scenarios = np.sort(scenarios, axis=0)
    sorted_diff = sorted_scenarios - actual_ndarray

    H = np.where(sorted_diff<0, 0, 1)

    F = np.tile(np.linspace(0, M, M).reshape([M,1])/M, [1, N])

    return np.sum((H-F)**2, axis = 0)

def point_crps(actual_ndarray, scenarios):

    N = len(actual_ndarray)
    M = scenarios.shape[0]

    sorted_scenarios = np.sort(scenarios, axis=0)
    sorted_diff = sorted_scenarios - actual_ndarray

    H = np.where(sorted_diff<0, 0, 1)

    F = np.tile(np.linspace(0, M, M).reshape([M,1])/M, [1, N])

    first_term = (1-H)*F+H*(1-F)
    second_term = np.abs(sorted_diff)

    return 2 * np.mean(first_term*second_term, axis = 0)

def total_energy_score(actual_ts, ts_array, beta=1):

    first_term  =  np.mean(np.sqrt(np.sum((ts_array - actual_ts)**2, axis=1))**beta)

    # Build second term

    second_term = np.zeros(len(actual_ts))

    for i in ts_array:
        for j in ts_array:
            second_term += np.sqrt(np.sum((i - j)**2, axis=1))**beta

    second_term /= ts_array.shape[0]**2

    return first_term - 0.5 * second_term



def forecast_performance_per_asset(models_dict, actual_df, all_asset_names, bootstrap_list)->pd.DataFrame:
    # Initialize list to store the final results
    results = {}
    results_in_lists ={}

    list_number = len(bootstrap_list)
    sqrt_number_list = np.sqrt(list_number)
    
    # Loop over each asset in the list
    for asset_name in tqdm(all_asset_names):

        benchmark_single_ts_rmse=0
        benchmark_point_mae=0
        benchmark_point_crps=0
        benchmark_variogram_score_per_asset_05=0
        benchmark_variogram_score_per_asset_2=0

        results[asset_name] = []
        results_in_lists[asset_name] = []
        # Loop over each model in the dictionary
        for model_name, model_df in models_dict.items():
            # Initialize dictionaries to store the metrics for this model
            metrics = {"Models": model_name}
            metrics_in_lists = {"Models": model_name}
            
            # Get forecast data for the current asset
            forecast_data_full = model_df[model_df['Asset'] == asset_name].iloc[:, 2:].to_numpy().squeeze()
            actual_data = actual_df[actual_df['Asset'] == asset_name].iloc[:, 1:].to_numpy().squeeze()

            list_single_ts_rmse = np.zeros(list_number)
            list_variogram_score_per_asset_05 = np.zeros(list_number)  
            list_variogram_score_per_asset_2 = np.zeros(list_number)     
            list_ratio_single_ts_rmse = np.zeros(list_number)      
            list_ratio_variogram_score_per_asset_05 = np.zeros(list_number)   
            list_ratio_variogram_score_per_asset_2 = np.zeros(list_number)    
            list_mean_ratio_point_mae = np.zeros(list_number) 
            list_mean_ratio_point_crps = np.zeros(list_number)
            
            for index, subset in enumerate(bootstrap_list):
                
                forecast_data = forecast_data_full[subset]

                # Calculate mean forecast across scenarios
                forecast_mean = np.mean(forecast_data, axis=0).squeeze()

                if model_name==benchmark_name:

                    benchmark_single_ts_rmse = single_ts_rmse(actual_data, forecast_mean)
                    benchmark_point_mae = this_point_mae = point_mae(actual_data, forecast_data)
                    benchmark_point_crps = this_point_crps = point_crps(actual_data, forecast_data)

                    benchmark_variogram_score_per_asset_05 = variogram_score_per_asset(actual_data, forecast_data, p=0.5)
                    benchmark_variogram_score_per_asset_2 = variogram_score_per_asset(actual_data, forecast_data, p=2)

                    list_single_ts_rmse[index]     =   benchmark_single_ts_rmse
                    list_variogram_score_per_asset_05[index]       =   benchmark_variogram_score_per_asset_05
                    list_variogram_score_per_asset_2[index]        =   benchmark_variogram_score_per_asset_2

                else:

                    list_single_ts_rmse[index]       =  single_ts_rmse(actual_data, forecast_mean) 

                    this_point_mae        =  point_mae(actual_data, forecast_data) 
                    this_point_crps       =  point_crps(actual_data, forecast_data)

                    list_variogram_score_per_asset_05[index]     =  variogram_score_per_asset(actual_data, forecast_data, p=0.5) 
                    list_variogram_score_per_asset_2[index]      =  variogram_score_per_asset(actual_data, forecast_data, p=2)



                list_ratio_single_ts_rmse[index]       = list_single_ts_rmse[index]/ benchmark_single_ts_rmse 
                list_ratio_variogram_score_per_asset_05[index]     =  list_variogram_score_per_asset_05[index]/ benchmark_variogram_score_per_asset_05
                list_ratio_variogram_score_per_asset_2[index]      =  list_variogram_score_per_asset_2[index]/ benchmark_variogram_score_per_asset_2
                list_mean_ratio_point_mae[index]                 =   gmean(this_point_mae/benchmark_point_mae)
                list_mean_ratio_point_crps[index]                =   gmean(this_point_crps/benchmark_point_crps)

                # Append metrics for this model to the results list


            metrics["estimate_single_ts_rmse"] = np.mean( list_single_ts_rmse)
            metrics["estimate_variogram_score_per_asset_05"] = np.mean( list_variogram_score_per_asset_05)
            metrics["estimate_variogram_score_per_asset_2"] = np.mean( list_variogram_score_per_asset_2)
            metrics["estimate_ratio_single_ts_rmse"] = np.mean( list_ratio_single_ts_rmse)
            metrics["estimate_ratio_variogram_score_per_asset_05"] = np.mean( list_ratio_variogram_score_per_asset_05)
            metrics["estimate_ratio_variogram_score_per_asset_2"] = np.mean( list_ratio_variogram_score_per_asset_2)
            metrics["estimate_mean_ratio_point_mae"] = np.mean( list_mean_ratio_point_mae)
            metrics["estimate_mean_ratio_point_crps"] = np.mean( list_mean_ratio_point_crps)

            metrics["estimator_std_single_ts_rmse"] = np.std(list_single_ts_rmse)/ sqrt_number_list
            metrics["estimator_std_variogram_score_per_asset_05"] = np.std(list_variogram_score_per_asset_05)/ sqrt_number_list
            metrics["estimator_std_variogram_score_per_asset_2"] = np.std(list_variogram_score_per_asset_2)/ sqrt_number_list
            metrics["estimator_std_ratio_single_ts_rmse"] = np.std(list_ratio_single_ts_rmse)/ sqrt_number_list
            metrics["estimator_std_ratio_variogram_score_per_asset_05"] = np.std(list_ratio_variogram_score_per_asset_05)/ sqrt_number_list
            metrics["estimator_std_ratio_variogram_score_per_asset_2"] = np.std(list_ratio_variogram_score_per_asset_2)/ sqrt_number_list
            metrics["estimator_std_mean_ratio_point_mae"] = np.std(list_mean_ratio_point_mae)/ sqrt_number_list
            metrics["estimator_std_mean_ratio_point_crps"] = np.std(list_mean_ratio_point_crps)/ sqrt_number_list

            metrics_in_lists["single_ts_rmse"] = list_single_ts_rmse
            metrics_in_lists["variogram_score_per_asset_05"] = list_variogram_score_per_asset_05
            metrics_in_lists["variogram_score_per_asset_2"] = list_variogram_score_per_asset_2
            metrics_in_lists["ratio_single_ts_rmse"] = list_ratio_single_ts_rmse
            metrics_in_lists["ratio_variogram_score_per_asset_05"] = list_ratio_variogram_score_per_asset_05
            metrics_in_lists["ratio_variogram_score_per_asset_2"] = list_ratio_variogram_score_per_asset_2
            metrics_in_lists["mean_ratio_point_mae"] = list_mean_ratio_point_mae
            metrics_in_lists["mean_ratio_point_crps"] = list_mean_ratio_point_crps
            
            
            results[asset_name].append(metrics)
            results_in_lists[asset_name].append(metrics_in_lists)
        
    for asset_name in tqdm(all_asset_names):

        results[asset_name] = pd.DataFrame(results[asset_name])
        results_in_lists[asset_name] = pd.DataFrame(results_in_lists[asset_name])

    return results, results_in_lists

def plop_dict(d, key):
    return {key: d.pop(key)} | d

def group_assets_by_class(assets):
    asset_dict = defaultdict(list)
    
    for asset in assets:
        # Split by the first underscore
        class_name, asset_detail = asset.split('_', 1)
        asset_dict[class_name].append(asset)
    
    return dict(asset_dict)


def geometric_mean_of_dataframes(df_list):
    # Check if the list is empty
    if not df_list:
        raise ValueError("The list of DataFrames is empty.")
    
    # Get the columns from the first DataFrame
    cols = df_list[0].columns.tolist()
    
    # Ensure 'Models' is in the columns
    if 'Models' not in cols:
        raise ValueError("'Models' column is missing from the DataFrames.")
    
    # Extract the 'Models' column
    models = df_list[0]['Models']
    
    # Create a new DataFrame to store the results
    result_df = pd.DataFrame(columns=cols)
    result_df['Models'] = models
    
    # Iterate over the columns (except 'Models')
    for col in cols:
        if col != 'Models':
            # For each row, gather the lists of floats from all DataFrames
            geo_mean_values = []
            for idx in range(len(models)):
                # Extract the list of floats for the current row and column from all DataFrames
                lists_of_floats = [df[col].iloc[idx] for df in df_list]
                # Compute the geometric mean for each element in the list
                geo_mean_list = gmean(np.array(lists_of_floats), axis=0)
                geo_mean_values.append(geo_mean_list)

            # Add the new column with geometric mean lists to the result DataFrame
            result_df[col] = geo_mean_values

    return result_df

def compute_statistics(df):
    # Ensure the 'Models' column is present
    if 'Models' not in df.columns:
        raise ValueError("'Models' column is missing from the DataFrame.")

    # Get the column names (excluding 'Models')
    cols = [col for col in df.columns if col != 'Models']

    # Create a new DataFrame to store the results
    result_df = pd.DataFrame(columns=['Models'] + 
                                      [f'estimator_{col}' for col in cols] + 
                                      [f'std_{col}' for col in cols])
    result_df['Models'] = df['Models']

    # Compute the mean and standard deviation for each column
    for col in cols:
        means = []
        stds = []
        
        for idx in range(len(df)):
            lists_of_floats = df[col].iloc[idx]
            mean_value = np.mean(lists_of_floats)
            std_value = np.std(lists_of_floats) / np.sqrt(len(lists_of_floats))
            
            means.append(mean_value)
            stds.append(std_value)
        
        result_df[f'estimator_{col}'] = means
        result_df[f'std_{col}'] = stds

    return result_df

def main():

    testing_dates = list(testing_df.columns)
    my_index_start = testing_dates.index(starting_date_sim)
    my_index_end = testing_dates.index(end_date_test)+1
    testing_dates = testing_dates[my_index_start:my_index_end]

    bootstrap_list = np.zeros([bootstrap_number, number_of_paths])
    bootstrap_list[0,:] = np.arange(number_of_paths)

    if bootstrap_number>1:
        bootstrap_list[1:,:] = np.random.randint(number_of_paths, size = [bootstrap_number-1, number_of_paths])

    bootstrap_list = bootstrap_list.astype(int)

    actual_df = testing_df[testing_df["Asset"].isin(testing_assets_names_all)][["Asset"]+ testing_dates]

    model_dict={}
    folder_path = directory+"Forecasts_Analysis\\For_BT"
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            
            if "RW" in filename:
                model_name="RW"
            else:

                model_name = ""

                this_prior = None
                
                for prior in prior_name:
                    if prior in filename:

                        this_prior=prior

                for factors in factors_from:
                    if factors in filename:

                        this_factor = factors

                if this_prior is None:
                    model_name+= "VAR_"+this_factor
                else:
                    model_name+= "BVAR_"+this_factor+"_"+this_prior

                if "[2, 1, 1]" in filename:
                    model_name+= "_[2, 1, 1]"

            df = adjust_base_df(pd.read_csv(file_path, sep=";"))[["Asset", "Scenario"]+ testing_dates]

            model_dict[model_name]=df[df["Asset"].isin(testing_assets_names_all)]


    model_dict = plop_dict(model_dict, benchmark_name)

    print("Finished building dicts.")

    # Let's get the assets classes lists

    class_dict = group_assets_by_class(testing_assets_names_all)

    class_dict["All_300"] = testing_assets_names_all
    class_dict["Elite_20"] = testing_assets_names_elite

    result_to_export, result_in_lists = forecast_performance_per_asset(model_dict, actual_df, all_asset_names=testing_assets_names_all, bootstrap_list=bootstrap_list)

    for key, value in class_dict.items():

        df_list = [result_in_lists[asset_name] for asset_name in value]

        result_to_export[f"AGGREGATED_{key}"] = compute_statistics(geometric_mean_of_dataframes(df_list))



    # Export

    excel_path = directory+f"Forecasts_Analysis\\test_forecasts_(window {starting_date_sim.replace("/", "_")} to {end_date_test.replace("/", "_")}).xlsx"

    base = pd.DataFrame({"Assets": testing_assets_names_all})

    base.to_excel(excel_path)

    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:

        # Loop through the dictionary and write each DataFrame to a sheet
        for sheet_name, df in result_to_export.items():
            df.to_excel(writer, sheet_name=sheet_name[:30], index=False)

#RUN!

if __name__ == "__main__":

    main()




