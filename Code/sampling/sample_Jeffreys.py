import sys
import os
sys.path.append(os.path.abspath(r"C:\Users\valeriiz\OneDrive - ORTEC Finance\Desktop\Thesis\Code"))
from pathlib import Path
import numpy as np
import pandas as pd
import time
from Methods.variable_models import Model, key_word_dict
from Methods.utils.priors import Jeffreys, vectorize_matrix, vectorize_parameters, export_samples, make_positive_semidefinite

tails_external_sides = [500,500]
number_of_paths = 2000
starting_dates = ["12/2012", "12/2014", "12/2016", "12/2018", "12/2020", "12/2022"]
factors_from = "FSA"
prior_name = "Jeffreys_prior"
number_of_lags_list=[1]

directory = "C:\\Users\\valeriiz\\OneDrive - ORTEC Finance\\Desktop\\Thesis\\Data\\priors\\"
frequencies = ["Trend", "Business cicle", "Monthly cicle"]

priors ={}


for number_of_lags in number_of_lags_list:
    for frequency in frequencies:

        prior = Jeffreys()

        priors[frequency+str(number_of_lags)] = prior

initial_start_time =time.time()
df=pd.read_excel(r"C:\Users\valeriiz\OneDrive - ORTEC Finance\Desktop\Thesis\Data\Thesis data set - extended data.xlsx", sheet_name="Final Data")

for starting_date in starting_dates:

    my_model=Model(df, start=starting_date)
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

    components = {"Trend": 3,"Business cicle": 9,"Monthly cicle": 10 }
    jumps = { "Trend": 96,"Business cicle": 12,"Monthly cicle": 1 }

    if starting_date is not None:

        starting_date = starting_date.replace("/","_")



    for number_of_lags in number_of_lags_list:

        

        for frequency in frequencies:

            folder_name = f"F_{frequency}_L_{number_of_lags}_f_{factors_from}_w_{number_of_paths}_d_{starting_date}"

            data = my_model.decomposed_dict[factors_from][frequency][key_word_dict[factors_from]][:components[frequency]]

            priors[frequency+str(number_of_lags)].aggregate_likelihood(data, number_of_lags= number_of_lags, jumps= jumps[frequency])

            sample_dict = priors[frequency+str(number_of_lags)].sample(number_of_paths)

            print(f"Done for {frequency} and {number_of_lags}")

            loc = directory + prior_name+f"\\samples\\{folder_name}"

            Path(loc).mkdir(parents=True, exist_ok=True) 

            export_samples(loc, sample_dict)