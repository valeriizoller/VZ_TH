import sys
import os
sys.path.append(os.path.abspath(r"C:\Users\valeriiz\OneDrive - ORTEC Finance\Desktop\Thesis\Code"))

import numpy as np
import pandas as pd
import time
from Methods.variable_models import Model, key_word_dict
from Methods.utils.priors import read_samples
tails_external_sides=[500,500]


number_of_paths=2000
number_of_paths_to_test= 2000
starting_date = None #"12/2010"
factors_from = "FSA"
priors_names= ["Conj_Norm_Inv_Wish_Prior"] #["Zellners_prior", "Jeffreys_prior", "Minnesota_prior", "uniform_prior", "Ind_Norm_Inv_Wish_Prior", "Conj_Norm_Inv_Wish_Prior"]
number_of_lags=[1,2,3]

directory = "C:\\Users\\valeriiz\\OneDrive - ORTEC Finance\\Desktop\\Thesis\\Data\\priors\\"
output_directory = "C:\\Users\\valeriiz\\OneDrive - ORTEC Finance\\Desktop\\Thesis\\Results\\Tests\\"
frequencies = ["Trend", "Business cicle", "Monthly cicle"]

initial_start_time =time.time()
df=pd.read_excel(r"C:\Users\valeriiz\OneDrive - ORTEC Finance\Desktop\Thesis\Data\Thesis data set - extended data.xlsx", sheet_name="Final Data")

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

for prior_name in priors_names:
    test_data = pd.DataFrame(columns=["Model", "Prior","Frequency", "Lags", "Factors"])

    for number_of_lags_i in number_of_lags:
        
        parameters_dict={}
        lags={}

        for frequency in frequencies:

            folder_name = f"F_{frequency}_L_{number_of_lags_i}_f_{factors_from}_w_{number_of_paths}_d_{starting_date}"
            
            loc_name = directory + prior_name+f"\\samples\\{folder_name}"

            parameters_dict[frequency] = read_samples(loc_name, components[frequency], number_of_lags_i)
            lags[frequency] = number_of_lags_i

        my_model.build_BVAR(parameters_dict, components, jumps, lags, components_from = factors_from)

        for frequency in frequencies:
            this_freq_dict=my_model.processes[frequency].get_information_criterias(number_of_paths=number_of_paths_to_test)
            this_freq_dict_2=my_model.processes[frequency].get_error_tests(number_of_paths=number_of_paths_to_test)

            this_freq_dict = this_freq_dict | this_freq_dict_2

            this_freq_dict["Model"]=f"{prior_name}_{frequency}_ {number_of_lags_i} lags _{factors_from}"
            this_freq_dict["Prior"]= prior_name
            this_freq_dict["Frequency"]=frequency
            this_freq_dict["Lags"]=number_of_lags_i
            this_freq_dict["Factors"]= factors_from 

            df = pd.DataFrame(this_freq_dict, index=[0])

            test_data = pd.concat([test_data, df], ignore_index=True)

        print(f"Done {number_of_lags_i}!")

    test_data.to_excel(output_directory+f"\\BVAR_{prior_name}_ {number_of_lags} lags _{factors_from}.xlsx")

