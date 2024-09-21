import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os
sys.path.append(os.path.abspath(r"C:\Users\valeriiz\OneDrive - ORTEC Finance\Desktop\Thesis\Code"))
from Methods.variable_models import *

initial_start_time =time.time()
number_of_paths=2000
maturity_in_trend_steps=2
starting_date_sim_s = ["12/2022","12/2020","12/2018","12/2016","12/2014","12/2012","12/2010"]


df=pd.read_excel(r"C:\Users\valeriiz\OneDrive - ORTEC Finance\Desktop\Thesis\Data\Thesis data set - extended data.xlsx", sheet_name="Final Data")


for starting_date_sim in starting_date_sim_s:
    ending_date_cal = starting_date_sim
    my_model=Model(df, start=starting_date_sim)
    number_of_dimensions=df.index.to_list()[-1]+1
    print(f"num of variables {number_of_dimensions} and forecast starts at {my_model.current_date}. Import took {time.time()-initial_start_time}.")


    my_model.asset_map = my_model.df_initial["Asset"].to_dict()

    my_model.variables_df_time_series = my_model.df_initial

    my_model.variables_forecast = np.zeros([number_of_paths, number_of_dimensions, maturity_in_trend_steps*96])

    data = my_model.variables_df_time_series.iloc[:, 2:].to_numpy()

    my_model.variables_forecast[:, :, 0]=np.tile(data[:, -1].reshape([1,number_of_dimensions]), [number_of_paths,1])

    row_means, covariance_matrix = vectorized_rowwise_mean_cov(data[:,1:]-data[:,:-1])

    steps = np.random.multivariate_normal(row_means, covariance_matrix, size=[maturity_in_trend_steps*96-1, number_of_paths])

    for t in range(maturity_in_trend_steps*96-1):
        
        my_model.variables_forecast[:, :, t+1] = my_model.variables_forecast[:, :, t]+steps[t, :]

    start_time =time.time()
    print("Exporting forecasts...", end="\r")
    my_model.export_forecasts("C:\\Users\\valeriiz\\OneDrive - ORTEC Finance\\Desktop\\Thesis\\Results\\Forecasts\\", additional=["RW", starting_date_sim.replace("/", "_"), ending_date_cal.replace("/", "_")], max_date="12/2024")
    print(f"Exported forecasts. ({time.time()-start_time})")


    print(f"max_error: {np.max(np.nan_to_num(np.abs(my_model.df_initial.iloc[:,2:].to_numpy()-my_model.variables_df_time_series.iloc[:,2:].to_numpy())))}")

    for i in np.random.randint(low=0, high=300, size=5):
        my_model.display_variables_forecast(int(i), distribution=True, also_frequencies=False)

    my_model.display_variables_forecast(174, distribution=True, also_frequencies=False)
    my_model.display_variables_forecast(166, distribution=True, also_frequencies=False)
    my_model.display_variables_forecast(145, distribution=True, also_frequencies=False)
    my_model.display_variables_forecast(44, distribution=True, also_frequencies=False)


    print(f"Total time: {time.time()-initial_start_time}")