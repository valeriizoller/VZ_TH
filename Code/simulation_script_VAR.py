import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os
sys.path.append(os.path.abspath(r"C:\Users\valeriiz\OneDrive - ORTEC Finance\Desktop\Thesis\Code"))
from Methods.variable_models import *

initial_start_time =time.time()
tails_external_sides=[500,500]
number_of_paths=2000
maturity_in_trend_steps=5
starting_date_sim = "12/2022"
ending_date_cal = starting_date_sim
factors_from_s = ["FSA"]

components = {"Trend": 3,"Business cicle": 9,"Monthly cicle": 10 }
jumps = { "Trend": 96,"Business cicle": 12,"Monthly cicle": 1 }
lags = { "Trend": 2,"Business cicle": 1,"Monthly cicle": 1 }

df=pd.read_excel(r"C:\Users\valeriiz\OneDrive - ORTEC Finance\Desktop\Thesis\Data\Thesis data set - extended data.xlsx", sheet_name="Final Data")

my_model=Model(df, start=starting_date_sim)
number_of_dimensions=df.index.to_list()[-1]+1
print(f"num of variables {number_of_dimensions} and forecast starts at {my_model.current_date}. Import took {time.time()-initial_start_time}.")

start_time =time.time()
print("Elaborating data...", end="\r")
my_model.elaborate_data_new_1()
print(f"Elaborating data. ({time.time()-start_time})")

start_time =time.time()
print("Decomposing into 3 frequencies...", end="\r")
my_model.decompose(tails_external_sides=tails_external_sides)
print(f"Decomposing into 3 frequencies. ({time.time()-start_time})")

start_time =time.time()
print("Elaborating data...", end="\r")
my_model.elaborate_data_2()
print(f"Elaborating data. ({time.time()-start_time})")

start_time =time.time()
print("Standardizing components...", end="\r")
my_model.standardize_decomposed_dict("Decomposed df LogReturns_F")
print(f"Standardizing components. ({time.time()-start_time})")

start_time =time.time()
factors = {}

for f_and_t in my_model.frequency_names:
    
    names=pd.read_excel(f"C:\\Users\\valeriiz\\OneDrive - ORTEC Finance\\Desktop\\Thesis\\Data\\Factors\\O_FSA_Factors_{f_and_t}.xlsx", sheet_name="Factors")["Factor"].to_list()

    factors[f_and_t]=my_model.decomposed_dict["DR_input"][f_and_t].copy().rename(columns={"Asset": "Factor"}).drop(columns=["Datashape"])

    factors[f_and_t]=factors[f_and_t][factors[f_and_t]["Factor"].isin(names)]

    factors[f_and_t] = factors[f_and_t].dropna(how="any", axis=0)

    factors[f_and_t]['Order'] = factors[f_and_t]['Factor'].map({name: i for i, name in enumerate(names)})

    factors[f_and_t] = factors[f_and_t].sort_values('Order').drop('Order', axis=1)
    
    factors[f_and_t].reset_index(drop=True, inplace=True)

print("Reducing dimensionality...", end="\r")
my_model.do_FSA(factors_pool=factors, max_expl=10, fixed_order=True)
my_model.do_PCA()
print(f"Reducing dimensionality. ({time.time()-start_time})")

for factors_from in factors_from_s:
        
    start_time =time.time()
    print("Building core model...", end="\r")
    my_model.build_VAR(
        estimator="YW",
        components=components,
        jumps=jumps,
        lags=lags,
        components_from= factors_from
    )
    print(f"Building core model. ({time.time()-start_time})")

    start_time =time.time()
    print("Forecasting...", end="\r")
    my_model.forecast(number_of_paths, maturity_in_trend_steps=maturity_in_trend_steps)
    print(f"Forecasting. ({time.time()-start_time})")

    start_time =time.time()
    print("Getting regressors...", end="\r")
    my_model.get_transformation_regressors()
    print(f"Getting regressors. ({time.time()-start_time})")

    start_time =time.time()
    print("Factors to variables...", end="\r")
    my_model.factors_to_standardized_variables()
    print(f"Factors to variables. ({time.time()-start_time})")

    start_time =time.time()
    print("Destandardizing...", end="\r")
    my_model.destandardize_variables_2()
    print(f"Destandardizing. ({time.time()-start_time})")

    start_time =time.time()
    print("Modeling residuals...", end="\r")
    my_model.model_residual()
    print(f"max_error: {np.max(np.nan_to_num(np.abs(my_model.decomposed_variables_df_time_series['Trend'].iloc[:,3:].to_numpy()-my_model.decomposed_dict['Decomposed df LogReturns_F']['Trend'].iloc[:,3:].to_numpy())))}")
    print(f"Modeling residuals. ({time.time()-start_time})")

    start_time =time.time()
    print("Elaborating forecasts...", end="\r")
    my_model.elaborate_forecasts()
    print(f"Elaborating forecasts. ({time.time()-start_time})")

    start_time =time.time()
    frequency_to_display="Trend"

    print("Aggregating frequencies...", end="\r")
    my_model.aggregate_frequencies()
    my_model.df_elaborated_1.iloc[:3, :]
    my_model.variables_df_time_series_in_LogIndex.iloc[:3, :]
    print(f"Aggregating frequencies. ({time.time()-start_time})")

    start_time =time.time()
    index_to_display =int(np.random.randint(low=0, high=300))

    # for i in range(np.min([number_of_paths, 100])):
    #     plt.plot(my_model.variables_forecast_in_LogIndex[i,index_to_display,:], color="blue")
    # plt.show()

    print("Elaborating forecasts...", end="\r")
    my_model.elaborate_forecast_2(keep_logindex=True)
    print(f"Elaborating forecasts. ({time.time()-start_time})")

    start_time =time.time()
    print("Exporting forecasts...", end="\r")
    my_model.export_forecasts("C:\\Users\\valeriiz\\OneDrive - ORTEC Finance\\Desktop\\Thesis\\Results\\Forecasts\\", additional=[starting_date_sim.replace("/", "_"), factors_from, ending_date_cal.replace("/", "_"), list(lags.values())], max_date="12/2024")
    print(f"Exported forecasts. ({time.time()-start_time})")


    print(f"max_error: {np.max(np.nan_to_num(np.abs(my_model.df_initial.iloc[:,2:].to_numpy()-my_model.variables_df_time_series.iloc[:,2:].to_numpy())))}")

    for i in np.random.randint(low=0, high=300, size=5):
        my_model.display_variables_forecast(int(i), distribution=True)

    my_model.display_variables_forecast(174, distribution=True)
    my_model.display_variables_forecast(166, distribution=True)
    my_model.display_variables_forecast(145, distribution=True)
    my_model.display_variables_forecast(44, distribution=True)


    print(f"Total time: {time.time()-initial_start_time}")