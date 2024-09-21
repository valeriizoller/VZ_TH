import pandas as pd
import numpy as np


def asset_type_map(x: str):
    return x.split("_")[0]

def RV_map(x: str):
    if x.split("_")[-1]=="RV":
        return "RV"
    else:
        return "Values"

Countries_map={"Australia": "Oceania", 
               "Austria":"Europe", 
               "Belgium":"Europe", 
               "Canada": "North_America", 
               "Denmark":"Europe", 
               "Europe":"Europe", 
               "Finland":"Europe", 
               "France":"Europe", 
               "Germany":"Europe", 
               "Greece":"Europe", 
               "Ireland":"Europe", 
               "Italy":"Europe", 
               "Japan":None, 
               "Netherlands":"Europe", 
               "New_Zealand": "Oceania", 
               "Norway":"Europe", 
               "Portugal":"Europe", 
               "Spain":"Europe", 
               "Sweden":"Europe", 
               "Switzerland":"Europe", 
               "United_Kingdom":"Europe", 
               "United_States": "North_America",
               "Emerging_Markets":None,
               "World": None}

def asset_country_map(x: str):
    for country in list(Countries_map.keys()):
        if country in x:
            return country
    
    return None

def macro_region(x):
    if x is not None:
        return Countries_map[x]
    else:
        return None
 
#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################

def get_factor_df(variables: pd.DataFrame, dates: list, include: bool=True, weight_df: pd.DataFrame|None = None, shifts: list[int]|None=None)->pd.DataFrame|None:
    if include:
        factor_df=variables.copy().rename(columns={"Asset": "Factor"}).drop(columns=["Datashape"])
    else:
        factor_df=pd.DataFrame(columns=variables.columns).rename(columns={"Asset": "Factor"}).drop(columns=["Datashape"])

    if "frequency" in factor_df.columns:
        factor_df=factor_df.drop(columns=["frequency"])

    working_df=variables.copy()

    adjust_df=[working_df]

    if weight_df is not None:
        A=variables[["Asset"]+dates].set_index("Asset").astype(np.float64)
        B=weight_df[["Asset"]+dates].fillna(0).set_index("Asset").astype(np.float64)

        weighted_df=(A*B).reset_index()
        adjust_df.append(weighted_df)
        adjust_df.append(weight_df)

    for df in adjust_df:
        df["Asset_class"]       =       df["Asset"].map(asset_type_map)
        df["RV"]                =       df["Asset"].map(RV_map)
        df["Country"]           =       df["Asset"].map(asset_country_map)
        df["Region"]            =       df["Country"].map(macro_region)

    classifiers=['Asset_class', 'RV',  "Region"]

    classifier_dict={}

    for classifier in classifiers:
        classifier_dict[classifier]=list(set(working_df[classifier].to_list()))
    
    dim=len(classifiers)

    methods=[mean_aggregation, median_aggregation, weighted_mean_aggregation]

    input={}
    input["dates"]=dates
    input["working_df"]=working_df

    if weight_df is not None:
        input["weighted_df"]=weighted_df
        input["weight_df"]=weight_df



    for phase in range(2**dim):
        selection=bin(phase)[2:]
        combination=[]

        for index, value in enumerate(selection):
            if value=="1":
                new_index=dim-len(selection)-index
                combination.append(classifiers[new_index])

        input["combination"]=combination

        for method in methods:
            adding_df=method(input)

            if not adding_df.empty:
                if factor_df.empty:
                    factor_df=adding_df.loc[:,list(factor_df.columns)].reset_index(drop=True)
                else:
                    factor_df=pd.concat([factor_df, adding_df]).reset_index(drop=True).loc[:,list(factor_df.columns)]


    factor_df=factor_df.dropna().fillna(0).reset_index(drop=True).drop_duplicates(subset=dates)

    if isinstance(factor_df, pd.DataFrame):

        if shifts is not None:
            factor_df=shifts_func(factor_df, shifts)
            
        return factor_df
    else:
        return None


def mean_aggregation(input: dict)->pd.DataFrame:

    combination=input["combination"]
    working_df=input["working_df"]
    dates=input["dates"]


    if len(combination)==0:
        mean_df=working_df.loc[:,dates].mean(axis=0).to_frame().transpose()
        mean_df["Factor"]= "Mean"

        if not isinstance(mean_df, pd.DataFrame):
            raise Warning("Wtf, we have a "+str(mean_df.type()))
    else:
        mean_df=working_df.groupby(combination)[dates].mean().reset_index()
        mean_df["Factor"]=mean_df[combination].apply("_".join, axis=1)
        mean_df["Factor"]= "Mean_"+ mean_df["Factor"].astype(str)
        mean_df=mean_df.drop(columns=combination)

    return mean_df

def median_aggregation(input: dict)->pd.DataFrame:

    combination=input["combination"]
    working_df=input["working_df"]
    dates=input["dates"]


    if len(combination)==0:
        mean_df=working_df.loc[:,dates].median(axis=0).to_frame().transpose()
        mean_df["Factor"]= "Median"

        if not isinstance(mean_df, pd.DataFrame):
            raise Warning("Wtf, we have a "+str(mean_df.type()))
    else:
        mean_df=working_df.groupby(combination)[dates].median().reset_index()
        mean_df["Factor"]=mean_df[combination].apply("_".join, axis=1)
        mean_df["Factor"]= "Median_"+ mean_df["Factor"].astype(str)
        mean_df=mean_df.drop(columns=combination)

    return mean_df

def weighted_mean_aggregation(input: dict)->pd.DataFrame:

    combination=input["combination"]
    dates=input["dates"]
    weighted_df=input["weighted_df"]
    weight_df=input["weight_df"]

    weight_df[dates]=weight_df[dates].astype(np.float64)
    weighted_df[dates]=weighted_df[dates].astype(np.float64)


    if len(combination)==0:
        num_df=weighted_df.loc[:,dates].mean(axis=0).to_frame().transpose()
        den_df=weight_df.loc[:,dates].astype(np.float64).sum(axis=0).to_frame().transpose()

        num_df["Factor"]= "Weighted_Mean"
        den_df["Factor"]= "Weighted_Mean"

        num_df= num_df.set_index("Factor")
        den_df= den_df.set_index("Factor")

        mean_df=(num_df/den_df).reset_index()

        if not isinstance(mean_df, pd.DataFrame):
            raise Warning("Wtf, we have a "+str(mean_df.type()))
    else:
        num_df=weighted_df.groupby(combination)[dates].mean().reset_index()
        den_df=weight_df.groupby(combination)[dates].sum().reset_index()

        num_df["Factor"]=num_df[combination].apply("_".join, axis=1)
        num_df["Factor"]= "Weighted_Mean_"+ num_df["Factor"].astype(str)

        den_df["Factor"]=den_df[combination].apply("_".join, axis=1)
        den_df["Factor"]= "Weighted_Mean_"+ den_df["Factor"].astype(str)

        num_df=num_df.drop(columns=combination)
        den_df=den_df.drop(columns=combination)

        num_df= num_df.set_index("Factor")
        den_df= den_df.set_index("Factor")

        mean_df=(num_df.astype(np.float64)/den_df.astype(np.float64)).reset_index()


    return mean_df


def shifts_func(input_df: pd.DataFrame, shifts: list[int])->pd.DataFrame:

    list_of_df=[input_df]

    for shift in shifts:
        shifted_df = input_df.copy()
        shifted_df.iloc[:, 1:] = input_df.iloc[:, 1:].shift(shift, axis=1)

        shifted_df["Factor"] = input_df["Factor"] + "_shifted_by_["+str(shift)+"]"

        if shift>0:
            for i in range(1,shift+1):
                shifted_df.iloc[:, i] = input_df.iloc[:, 1]

        elif shift<0:
            for i in range(shift, 0):
                shifted_df.iloc[:, i] = input_df.iloc[:, -1]


        list_of_df.append(shifted_df)


    return pd.concat(list_of_df).reset_index(drop=True)


