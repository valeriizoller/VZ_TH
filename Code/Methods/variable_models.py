from .factor_models import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm as tqdm
from datetime import datetime
import calendar

key_word_dict={
    "FSA" : "Factors",
    "PCA" : "components"
}

jump_dict={
    "Trend": 96,
    "Business cicle": 12,
    "Monthly cicle": 1
}

start_dict={
    "Trend": "12/1908",
    "Business cicle": "12/1974",
    "Monthly cicle": "12/1995"
}

class Model:

    def __init__(self, df= None, start :str| None = None):

        self.frequency_names = ["Trend", "Business cicle", "Monthly cicle"]
        
        self.df_initial=adjust_base_df(df, start=start)

        self.current_date = self.df_initial.columns.to_list()[-1]
        self.variable_names = self.df_initial["Asset"].to_list()

        self.number_of_variables=self.df_initial.shape[0]

        self.jump_dict=jump_dict

        
    def elaborate_data_1(self, df_fake):

        self.df_elaborated_1=self.df_initial.copy()

        dates=list(self.df_elaborated_1)[2:]

        new_table = df_fake.loc[df_fake["Asset"].isin(list(self.df_elaborated_1[self.df_elaborated_1["Datashape"]=="LogReturn"]["Asset"])),dates[:12]].to_numpy()

        new_table=np.cumsum(new_table, axis=1)

        self.df_elaborated_1.loc[self.df_elaborated_1["Datashape"]=="LogReturn",dates[:12]]=new_table #df_fake.loc[df_fake["Asset"].isin(list(self.df_elaborated_1[self.df_elaborated_1["Datashape"]=="LogReturn"]["Asset"])),dates[:12]].to_numpy()

        for i, end_date in enumerate(dates[12:]):

            date=dates[i]

            self.df_elaborated_1.loc[self.df_elaborated_1["Datashape"]=="LogReturn",end_date]=self.df_elaborated_1.loc[self.df_elaborated_1["Datashape"]=="LogReturn", date].to_numpy()  +   self.df_initial.loc[self.df_initial["Datashape"]=="LogReturn", end_date].to_numpy()

        self.df_elaborated_1.loc[self.df_elaborated_1["Datashape"]=="LogReturn","Datashape"]="LogIndex"


    def elaborate_data_new_1(self, log_levels=True, extrapolate=False, flat=False):
        
        if log_levels:
            self.df_elaborated_1=self.df_initial.apply(levels_to_loglevels, axis=1)
        else:
            self.df_elaborated_1=self.df_initial.copy()

        if extrapolate:
            if flat:
                self.df_elaborated_1=self.df_elaborated_1.apply(extrapolate_flat_timeseries, axis=1)
            else:
                self.df_elaborated_1=self.df_elaborated_1.apply(extrapolate_timeseries, axis=1)


    def decompose(self, tails_external_sides, decomposition_intervals=[2,24,192,np.inf], decomposition_type="CF"):
        self.decomposed_dict=decompose_df(self.df_elaborated_1, decomposition_intervals=decomposition_intervals,  decomposition_type=decomposition_type, tails_external_sides=tails_external_sides)

        self.asset_map=self.decomposed_dict["Assets_dict"]
        self.inv_asset_map=invert_dictionary(self.asset_map)

    def elaborate_data_2(self):

        self.decomposed_dict["Decomposed df LogReturns_F"]={}

        for frequency in self.frequency_names:

            self.decomposed_dict["Decomposed df LogReturns_F"][frequency] = self.decomposed_dict["Decomposed df"][frequency].copy()

            self.decomposed_dict["Decomposed df LogReturns_F"][frequency] = self.decomposed_dict["Decomposed df LogReturns_F"][frequency].apply(log_index_to_log_returns, axis=1)#.dropna(axis=1, how="any")

            selected_columns = self.decomposed_dict["Decomposed df LogReturns_F"][frequency].columns.to_list()

            selected_index = selected_columns.index(start_dict[frequency])

            selected_columns=selected_columns[:3]+selected_columns[selected_index:]

            self.decomposed_dict["Decomposed df LogReturns_F"][frequency] = self.decomposed_dict["Decomposed df LogReturns_F"][frequency][selected_columns]

    def standardize_decomposed_dict(self, voice: str):

        self.decomposed_dict["Standardized dict from "+voice]={}
        self.decomposed_dict["Standardized "+voice]={}
        self.decomposed_dict["DR_input"]={}

        for frequency in self.frequency_names:

            dates=list(self.decomposed_dict[voice][frequency].columns)[3:]

            time_series = self.decomposed_dict[voice][frequency][dates].copy().to_numpy()

            standardized_dict = standardize_time_series(time_series)

            self.decomposed_dict["Standardized dict from "+voice][frequency]=standardized_dict

            new_df = self.decomposed_dict[voice][frequency].copy()

            new_df.loc[:,dates] = standardized_dict["Standardized_Time_Series"]

            self.decomposed_dict["Standardized "+voice][frequency]= new_df

            self.decomposed_dict["DR_input"][frequency]= new_df


    def do_PCA(self):
        PCA_on_decomposed_df(self.decomposed_dict)

    def do_FSA(self, factors_pool, max_expl:int, fixed_order:bool):
        FSA_on_decomposed_df(self.decomposed_dict, factors_pool=factors_pool, max_expl=max_expl, fixed_order=fixed_order)

    def build_VAR(self,estimator:str, components:dict, jumps:dict, lags: dict, components_from: str, Sigma_from_OLS:bool=True):
        
        self.base={}
        self.estimates={}
        self.processes={}
        self.components_dict=components
        self.jumps_dict=jumps
        self.lags_dict=lags


        for frequency_component in self.frequency_names:
            self.base[frequency_component]=self.decomposed_dict[components_from][frequency_component][key_word_dict[components_from]][:components[frequency_component]]
            
            if estimator=="YW":
                self.estimates[frequency_component] = YW_estimation(self.base[frequency_component], lags[frequency_component], jumps[frequency_component], sigma_from_OLS=Sigma_from_OLS)
                # print(self.estimates[frequency_component]["Sigma"])
            elif estimator=="OLS":
                self.estimates[frequency_component] = OLS_new_estimator(self.base[frequency_component], lags[frequency_component], jumps[frequency_component])
            

            self.processes[frequency_component]=VAR_process(number_of_lags=lags[frequency_component],
                                                            parameters=self.estimates[frequency_component]["Parameters"],
                                                            phi_0=self.estimates[frequency_component]["Phi_0"].squeeze(), 
                                                            time_series=self.base[frequency_component], 
                                                            covariance=self.estimates[frequency_component]["Sigma"],
                                                            jumps=jumps[frequency_component])

    def build_BVAR(self, parameters: dict, components:dict, jumps:dict, lags: dict,  components_from: str):
        
        self.base={}
        self.processes={}
        self.components_dict=components
        self.jumps_dict=jumps
        self.lags_dict=lags
        self.parameters=parameters


        for frequency_component in self.frequency_names:
            self.base[frequency_component]=self.decomposed_dict[components_from][frequency_component][key_word_dict[components_from]][:components[frequency_component]]

            self.processes[frequency_component]=BVAR_process(number_of_lags=lags[frequency_component], 
                                                             parameters=self.parameters[frequency_component]["Parameters"], 
                                                             phi_0=self.parameters[frequency_component]["Phi_0"], 
                                                             time_series=self.base[frequency_component], 
                                                             covariance=self.parameters[frequency_component]["Sigma"],
                                                             jumps=jumps[frequency_component])


    def forecast(self, number_of_paths:int, maturity_in_trend_steps:int, interpolation_method:str="third"):

        self.number_of_paths=number_of_paths

        self.forecast_info={"number_of_paths": number_of_paths,
                            "interpolation_method": interpolation_method}

        self.horizons={"Trend": maturity_in_trend_steps,
                 "Business cicle": int(round(maturity_in_trend_steps*(self.jumps_dict["Trend"]/self.jumps_dict["Business cicle"]))),
                 "Monthly cicle": int(round(maturity_in_trend_steps*(self.jumps_dict["Trend"]/self.jumps_dict["Monthly cicle"])))}
        
        for frequency_component in self.frequency_names:
            self.processes[frequency_component].forecast(time_steps=self.horizons[frequency_component], number_of_paths=number_of_paths, jumps=self.jumps_dict[frequency_component])
            self.processes[frequency_component].interpolate_forecasts(number_of_points=self.jumps_dict[frequency_component], method=interpolation_method)

            print(f"Forecasted {frequency_component}")
    
    def display_factors_forecast(self, frequency_to_display, index_to_display):

        fig, dis=plt.subplots(2)

        dis[0].plot(self.processes[frequency_to_display].time_series[index_to_display], color="black")
        dis[0].scatter(0, self.processes[frequency_to_display].time_series[index_to_display,0], color="black")
        for i in range(self.number_of_paths):
            dis[0].scatter(self.processes[frequency_to_display].forecasts_timeline, self.processes[frequency_to_display].forecasts[i, index_to_display], color="red", alpha=0.2)
            dis[0].plot(self.processes[frequency_to_display].forecasts_timeline, self.processes[frequency_to_display].forecasts[i, index_to_display], color="red", alpha=0.2)


        dis[1].plot(self.processes[frequency_to_display].time_series[index_to_display], color="black")
        dis[1].scatter(0, self.processes[frequency_to_display].time_series[index_to_display,0], color="black")
        for i in range(self.number_of_paths):
            dis[1].scatter(self.processes[frequency_to_display].forecasts_timeline, self.processes[frequency_to_display].forecasts[i, index_to_display], color="red", alpha=0.2)
            dis[1].plot(self.processes[frequency_to_display].interpolated_forecasts_timeline, self.processes[frequency_to_display].interpolated_forecast[i,index_to_display], color="red", alpha=0.2)

        fig.suptitle(f'Component {index_to_display} of {frequency_to_display}', fontsize=16)

        plt.show()

    def get_transformation_regressors(self):

        self.transformation = {}

        for frequency_component in self.frequency_names:
            
            Y = self.decomposed_dict["DR_input"][frequency_component].to_numpy()[:,3:]
            X = self.base[frequency_component]

            self.transformation[frequency_component] = multilinear_multivariate_regression(X, Y)



    def factors_to_standardized_variables(self):

        self.variables_forecast_dict={}
        self.variables_base_dict={}

        self.variables_forecast_dict["Standardized Variables in LogReturns_F"]={}
        self.variables_base_dict["Standardized Variables in LogReturns_F"]={}
        self.variables_base_dict["df Standardized Variables in LogReturns_F"]={}

        for frequency_component in self.frequency_names:

            self.variables_forecast_dict["Standardized Variables in LogReturns_F"][frequency_component]=np.matmul(self.transformation[frequency_component], self.processes[frequency_component].interpolated_forecast)
            self.variables_base_dict["Standardized Variables in LogReturns_F"][frequency_component]=np.matmul(self.transformation[frequency_component], self.base[frequency_component])


            dates  = list(self.decomposed_dict["DR_input"][frequency_component])[3:]
            new_df = self.decomposed_dict["DR_input"][frequency_component].copy()

            new_df[dates]=self.variables_base_dict["Standardized Variables in LogReturns_F"][frequency_component]

            self.variables_base_dict["df Standardized Variables in LogReturns_F"][frequency_component]=new_df
    
    def destandardize_variables(self):

        self.variables_forecast_dict["Variables in LogReturns_F"]={}
        self.variables_base_dict["Variables in LogReturns_F"]={}
        self.variables_base_dict["df Variables in LogReturns_F"]={}

        for frequency_component in self.frequency_names:

            mean=self.decomposed_dict["Standardized dict from Decomposed df LogReturns_F"][frequency_component]["Mean"].reshape([1, self.number_of_variables, 1])
            std=self.decomposed_dict["Standardized dict from Decomposed df LogReturns_F"][frequency_component]["Std"].reshape([1, self.number_of_variables, 1])

            self.variables_forecast_dict["Variables in LogReturns_F"][frequency_component]   =   self.variables_forecast_dict["Standardized Variables in LogReturns_F"][frequency_component]*np.tile(std, [self.number_of_paths,1,self.horizons["Monthly cicle"]+1])+np.tile(mean, [self.number_of_paths,1,self.horizons["Monthly cicle"]+1])
            self.variables_base_dict["Variables in LogReturns_F"][frequency_component]       =   self.variables_base_dict["Standardized Variables in LogReturns_F"][frequency_component]*np.tile(std,[1,1,self.processes[frequency_component].date]).squeeze()+np.tile(mean,[1,1,self.processes[frequency_component].date]).squeeze()

            dates  = list(self.decomposed_dict["DR_input"][frequency_component])[3:]
            new_df = self.decomposed_dict["DR_input"][frequency_component].copy()

            new_df[dates]=self.variables_base_dict["Variables in LogReturns_F"][frequency_component]  

            self.variables_base_dict["df Variables in LogReturns_F"][frequency_component]=new_df

    def destandardize_variables_2(self):
        # Initialize dictionaries
        self.variables_forecast_dict["Variables in LogReturns_F"]={}
        self.variables_base_dict["Variables in LogReturns_F"]={}
        self.variables_base_dict["df Variables in LogReturns_F"]={}

        for frequency_component in self.frequency_names:

            mean=self.decomposed_dict["Standardized dict from Decomposed df LogReturns_F"][frequency_component]["Mean"].reshape([1, self.number_of_variables, 1])
            std=self.decomposed_dict["Standardized dict from Decomposed df LogReturns_F"][frequency_component]["Std"].reshape([1, self.number_of_variables, 1])

            mean_hist=self.decomposed_dict["Standardized dict from Decomposed df LogReturns_F"][frequency_component]["Mean"].reshape([self.number_of_variables, 1])
            std_hist=self.decomposed_dict["Standardized dict from Decomposed df LogReturns_F"][frequency_component]["Std"].reshape([self.number_of_variables, 1])

            self.variables_forecast_dict["Variables in LogReturns_F"][frequency_component]   =   self.variables_forecast_dict["Standardized Variables in LogReturns_F"][frequency_component]*std+mean
            self.variables_base_dict["Variables in LogReturns_F"][frequency_component]       =   self.variables_base_dict["Standardized Variables in LogReturns_F"][frequency_component]*std_hist+mean_hist

            dates  = list(self.decomposed_dict["DR_input"][frequency_component])[3:]
            new_df = self.decomposed_dict["DR_input"][frequency_component].copy()

            new_df[dates]=self.variables_base_dict["Variables in LogReturns_F"][frequency_component]  

            self.variables_base_dict["df Variables in LogReturns_F"][frequency_component]=new_df


    def model_residual(self, interpolation_method="third"):
        
        self.residual_df = {}

        self.residual_YW_estimates = {}

        self.residual_processes = {}

        self.decomposed_variables_forecasts={}

        self.decomposed_variables_df_time_series = self.variables_base_dict["df Variables in LogReturns_F"].copy()

        for frequency in self.frequency_names:

            self.residual_df[frequency] = self.variables_base_dict["df Variables in LogReturns_F"][frequency].copy()

            dates = list(self.residual_df[frequency].columns)[3:]

            self.residual_df[frequency][dates]=self.decomposed_dict["Decomposed df LogReturns_F"][frequency][dates].to_numpy()-self.variables_base_dict["df Variables in LogReturns_F"][frequency][dates].to_numpy()          
            
            self.residual_YW_estimates[frequency] = YW_residuals(self.residual_df[frequency][dates].to_numpy(), 1, self.jumps_dict[frequency])

            # CHANGE 2
            self.residual_processes[frequency]=VAR_process(number_of_lags=1, 
                                                           parameters=self.residual_YW_estimates[frequency]["Parameters"], 
                                                           phi_0=self.residual_YW_estimates[frequency]["Phi_0"].squeeze(), 
                                                           time_series=self.residual_df[frequency][dates].to_numpy(), 
                                                           covariance=self.residual_YW_estimates[frequency]["Sigma"],
                                                           jumps=self.jumps_dict[frequency],
                                                           diagonal=True)
            
            self.residual_processes[frequency].forecast(time_steps=self.horizons[frequency], number_of_paths=self.number_of_paths, jumps=self.jumps_dict[frequency])
            self.residual_processes[frequency].interpolate_forecasts(number_of_points=self.jumps_dict[frequency], method=interpolation_method)

            print(f"Forecasted {frequency}")

            self.decomposed_variables_forecasts[frequency] = self.residual_processes[frequency].interpolated_forecast + self.variables_forecast_dict["Variables in LogReturns_F"][frequency]

            self.decomposed_variables_df_time_series[frequency] = self.variables_base_dict["df Variables in LogReturns_F"][frequency].copy()

            self.decomposed_variables_df_time_series[frequency][dates]=self.residual_processes[frequency].time_series + self.variables_base_dict["df Variables in LogReturns_F"][frequency][dates].to_numpy()
            
    def elaborate_forecasts(self):

        self.decomposed_variables_forecast_in_LogIndex={}
        self.decomposed_variables_df_time_series_in_LogIndex = {}

        for frequency in self.frequency_names:
            
            self.decomposed_variables_df_time_series_in_LogIndex[frequency] = self.decomposed_dict["Decomposed df"][frequency].copy()

            # dates = list(self.decomposed_variables_df_time_series_in_LogIndex[frequency].columns)[3:]
            # return_dates = list(self.decomposed_dict["Decomposed df LogReturns_F"][frequency].columns)[3:]

            # selected_index=dates.index(start_dict[frequency])
            # dates=dates[selected_index-jump_dict[frequency]:]

            # print(frequency)
            # print(dates)
            # print(return_dates)

            # self.decomposed_variables_df_time_series_in_LogIndex[frequency].loc[:,dates]=np.zeros([self.number_of_variables, len(dates)])

            filter = (self.decomposed_variables_df_time_series_in_LogIndex[frequency]["Datashape"]=="LogIndex")

            positions_of_index = [index for index, value in enumerate(filter) if value==1]
            positions_of_level = [index for index, value in enumerate(filter) if value==0]

            # start = keep_first_n_non_nan(self.decomposed_variables_df_time_series_in_LogIndex[frequency].iloc[positions_of_index,3:].to_numpy(), len(dates)-len(return_dates))

            # for index, return_date in enumerate(return_dates):

            #     date = dates[index]
                
            #     self.decomposed_variables_df_time_series_in_LogIndex[frequency].loc[positions_of_index, return_date] = np.nan_to_num(self.decomposed_variables_df_time_series_in_LogIndex[frequency].loc[positions_of_index, date].to_numpy()) + np.nan_to_num(self.decomposed_variables_df_time_series[frequency].loc[positions_of_index, return_date].to_numpy())+start[:, index]
            
            # self.decomposed_variables_df_time_series_in_LogIndex[frequency].loc[positions_of_index, "Datashape"] = "LogIndex"

            # self.decomposed_variables_df_time_series_in_LogIndex[frequency] = get_nans_from(self.decomposed_variables_df_time_series_in_LogIndex[frequency], self.decomposed_dict["Decomposed df"][frequency])

            # self.decomposed_variables_df_time_series_in_LogIndex[frequency].loc[positions_of_level, return_dates] = self.decomposed_variables_df_time_series[frequency].loc[positions_of_level, return_dates].to_numpy().astype(np.float64)

            T= self.horizons["Monthly cicle"]

            S= self.jumps_dict[frequency]

            self.decomposed_variables_forecast_in_LogIndex[frequency]=np.zeros([self.number_of_paths, self.number_of_variables, T+S])

            self.decomposed_variables_forecast_in_LogIndex[frequency][:, positions_of_index, :S]= np.tile(self.decomposed_variables_df_time_series_in_LogIndex[frequency].iloc[positions_of_index, -S:].to_numpy(), [self.number_of_paths,1,1])

            for time_t in range(T):

                self.decomposed_variables_forecast_in_LogIndex[frequency][:,positions_of_index,S+time_t]=(self.decomposed_variables_forecast_in_LogIndex[frequency][:, positions_of_index, time_t] + self.decomposed_variables_forecasts[frequency][:, positions_of_index, time_t+1]).squeeze()

            self.decomposed_variables_forecast_in_LogIndex[frequency][:,positions_of_level,-T-1:] = self.decomposed_variables_forecasts[frequency][:,positions_of_level , -T-1:]


            self.decomposed_variables_forecast_in_LogIndex[frequency] = self.decomposed_variables_forecast_in_LogIndex[frequency][:,:,-T-1:]

    def aggregate_frequencies(self):

        T= self.horizons["Monthly cicle"]

        self.variables_forecast_in_LogIndex=np.zeros([self.number_of_paths, self.number_of_variables, T+1])
        self.variables_df_time_series_in_LogIndex = self.df_elaborated_1.copy()

        dates= list(self.variables_df_time_series_in_LogIndex)[2:]

        self.variables_df_time_series_in_LogIndex[dates]=np.zeros([self.number_of_variables, len(dates)])

        for frequency in self.frequency_names:
            self.variables_forecast_in_LogIndex  =  self.variables_forecast_in_LogIndex + self.decomposed_variables_forecast_in_LogIndex[frequency][:,:,-T-1:]
            self.variables_df_time_series_in_LogIndex[dates]  =  self.variables_df_time_series_in_LogIndex[dates].to_numpy() + self.decomposed_variables_df_time_series_in_LogIndex[frequency][dates].to_numpy()

    def elaborate_forecast_2(self, keep_logindex:bool):

        if keep_logindex:
            self.variables_df_time_series = self.variables_df_time_series_in_LogIndex
            self.variables_forecast = self.variables_forecast_in_LogIndex

        else:        
            T= self.horizons["Monthly cicle"]

            self.variables_df_time_series = self.df_elaborated_1.copy()
            self.variables_forecast = np.zeros([self.number_of_paths, self.number_of_variables, T+1])

            filter = (self.variables_df_time_series["Datashape"]=="LogReturn")

            positions_of_index = [index for index, value in enumerate(filter) if value==1]
            positions_of_level = [index for index, value in enumerate(filter) if value==0]

            dates = list(self.variables_df_time_series.columns)[2:]

            self.variables_df_time_series.loc[positions_of_index, dates[12:]] = self.variables_df_time_series_in_LogIndex.loc[positions_of_index, dates[12:]].to_numpy()-self.variables_df_time_series_in_LogIndex.loc[positions_of_index, dates[:-12]].to_numpy()

            self.variables_forecast[:,positions_of_index,:12] = self.variables_forecast_in_LogIndex[:,positions_of_index,:12]-np.tile(self.variables_df_time_series_in_LogIndex.loc[positions_of_index, dates[-12:]].to_numpy().reshape([1, len(positions_of_index), 12]), [self.number_of_paths, 1,1])

            self.variables_forecast[:,positions_of_index,12:] = self.variables_forecast_in_LogIndex[:,positions_of_index,12:] - self.variables_forecast_in_LogIndex[:,positions_of_index,:-12]

            self.variables_forecast[:,positions_of_level,:] = self.variables_forecast_in_LogIndex[:,positions_of_level,:]

        filter = (self.variables_df_time_series["Datashape"]=="LogLevel")

        self.variables_df_time_series=self.variables_df_time_series_in_LogIndex.apply(loglevels_to_levels, axis=1)

        positions_of_loglevels = [index for index, value in enumerate(filter) if value==1]

        self.variables_forecast[:, positions_of_loglevels, :] = forecast_loglevels_to_levels(self.variables_forecast[:, positions_of_loglevels, :])

    def export_forecasts(self, directory: str, additional: list|None=None, max_date=None):

        number_of_paths, number_of_variables, times =self.variables_forecast.shape

        start_date = datetime.strptime(self.current_date, "%m/%Y")

        def format_date_for_windows(date):
            return f"{date.month}/{date.year}"

        date_columns = [format_date_for_windows(start_date + pd.DateOffset(months=i)) for i in range(times)]

        if max_date is not None:
            date_max_index = date_columns.index(max_date)
            date_columns = date_columns[:date_max_index]

        # Reshape the array
        reshaped_data = np.transpose(self.variables_forecast, (1,0,2)).reshape(-1, times)[:, :len(date_columns)]

        # Create the list of variables and scenarios as per the new order
        variables = np.repeat(self.variable_names, number_of_paths)
        scenarios = np.tile(np.arange(number_of_paths), number_of_variables)

        # Create DataFrame with date-formatted columns
        df = pd.DataFrame(reshaped_data, columns=date_columns)
        df.insert(0, 'Scenario', scenarios)  # Insert 'Scenario' second
        df.insert(0, 'Variable', variables)  # Insert 'Variable' first

        # Get the current date and time in the desired format
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H_%M")

        cool = f"T_{times}_F_{number_of_paths}"

        if additional is not None:
            for add in additional:
                cool+= f"_{add}"

        # Create the filename with the datetime included
        filename = directory+f"forecast_{cool}_{current_datetime}.csv"

        # Export to CSV
        df.to_csv(filename, index=False, sep=";")



    def display_variables_forecast(self, input, distribution: bool, also_frequencies = True,  title_addition:None|str=None):
        if isinstance(input, int):
            index_to_display = input
            variable = self.asset_map[index_to_display]
        elif isinstance(input, str):
            variable = input
            index_to_display = self.inv_asset_map[variable]
        else:
            raise TypeError(f"input type has to be str or int, not {type(input)}")
        
        # Create subplots
        if also_frequencies:
        
            fig = make_subplots(rows=4, cols=1, subplot_titles=self.frequency_names + ["Total"], shared_xaxes=True, vertical_spacing=0.1)

            len_x=len(self.decomposed_dict["Decomposed df LogReturns_F"]["Trend"].iloc[index_to_display, 3:].to_numpy())+1+jump_dict["Trend"]
            x_vals = self.processes["Trend"].interpolated_forecasts_timeline + jump_dict["Trend"]

            for freq_index, frequency_to_display in enumerate(self.frequency_names):

                black_blue_x = np.arange(len_x-len(self.decomposed_dict["Decomposed df LogReturns_F"][frequency_to_display].iloc[index_to_display, 3:].to_numpy()),len_x)

                # Blue line
                fig.add_trace(
                    go.Scatter(
                        x=black_blue_x,
                        y=self.decomposed_dict["Decomposed df LogReturns_F"][frequency_to_display].iloc[index_to_display, 3:].to_numpy(),
                        mode='lines',
                        line=dict(color='blue', width=2),
                        opacity=0.5,
                        name=f'{frequency_to_display} - Decomposed'
                    ),
                    row=freq_index + 1, col=1
                )
                # Black line
                fig.add_trace(
                    go.Scatter(
                        x=black_blue_x,
                        y=self.decomposed_variables_df_time_series[frequency_to_display].iloc[index_to_display, 3:].to_numpy(),
                        mode='lines',
                        line=dict(color='black', width=2),
                        name=f'{frequency_to_display} - Variables'
                    ),
                    row=freq_index + 1, col=1
                )

                

                if distribution:
                    forecasts = self.decomposed_variables_forecasts[frequency_to_display][:, index_to_display, :]
                    mean_forecast = np.mean(forecasts, axis=0)
                    percentiles = np.percentile(forecasts, [0, 5, 25, 75, 95, 100], axis=0)

                    

                    # 0-100th percentile area
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([x_vals, x_vals[::-1]]),
                            y=np.concatenate([percentiles[5], percentiles[0][::-1]]),
                            fill='toself',
                            fillcolor='rgba(255, 0, 0, 0.1)',
                            line=dict(color='rgba(255, 0, 0, 0)'),
                            showlegend=False,
                            name=f'{frequency_to_display} - 0-100th Percentile'
                        ),
                        row=freq_index + 1, col=1
                    )

                    # 5-95th percentile area
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([x_vals, x_vals[::-1]]),
                            y=np.concatenate([percentiles[4], percentiles[1][::-1]]),
                            fill='toself',
                            fillcolor='rgba(255, 0, 0, 0.2)',
                            line=dict(color='rgba(255, 0, 0, 0)'),
                            showlegend=False,
                            name=f'{frequency_to_display} - 5-95th Percentile'
                        ),
                        row=freq_index + 1, col=1
                    )

                    # 25-75th percentile area
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([x_vals, x_vals[::-1]]),
                            y=np.concatenate([percentiles[3], percentiles[2][::-1]]),
                            fill='toself',
                            fillcolor='rgba(255, 0, 0, 0.4)',
                            line=dict(color='rgba(255, 0, 0, 0)'),
                            showlegend=False,
                            name=f'{frequency_to_display} - 25-75th Percentile'
                        ),
                        row=freq_index + 1, col=1
                    )

                    # Mean line
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=mean_forecast,
                            mode='lines',
                            line=dict(color='red', width=2),
                            name=f'{frequency_to_display} - Mean Forecast'
                        ),
                        row=freq_index + 1, col=1
                    )
                else:
                    # Red forecast lines
                    for i in range(self.number_of_paths):
                        fig.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=self.decomposed_variables_forecasts[frequency_to_display][i, index_to_display, :],
                                mode='lines',
                                line=dict(color='red', width=2),
                                opacity=0.2,
                                showlegend=False
                            ),
                            row=freq_index + 1, col=1
                        )


                fig.update_xaxes(title_text="time", range=[0, x_vals[-1]], row=freq_index + 1, col=1)

            black_x = np.arange(1,len_x+1)
            blue_x = np.arange(len_x-len(self.df_initial.iloc[index_to_display, 2:].to_numpy()), len_x)
            
            # Blue line for Total
            fig.add_trace(
                go.Scatter(
                    x= blue_x,
                    y=self.df_initial.iloc[index_to_display, 2:].to_numpy(),
                    mode='lines',
                    line=dict(color='blue', width=2),
                    opacity=0.5,
                    name='Total - Initial'
                ),
                row=4, col=1
            )
            # Black line for Total
            fig.add_trace(
                go.Scatter(
                    x=black_x,
                    y=self.variables_df_time_series.iloc[index_to_display, 2:].to_numpy(),
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Total - Variables'
                ),
                row=4, col=1
            )

            if distribution:
                forecasts = self.variables_forecast[:, index_to_display, :]
                mean_forecast = np.mean(forecasts, axis=0)
                percentiles = np.percentile(forecasts, [0, 5, 25, 75, 95, 100], axis=0)

                # x_vals = self.processes["Monthly cicle"].interpolated_forecasts_timeline+len_x

                # 0-100th percentile area
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_vals, x_vals[::-1]]),
                        y=np.concatenate([percentiles[5], percentiles[0][::-1]]),
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.1)',
                        line=dict(color='rgba(255, 0, 0, 0)'),
                        showlegend=False,
                        name='Total - 0-100th Percentile'
                    ),
                    row=4, col=1
                )

                # 5-95th percentile area
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_vals, x_vals[::-1]]),
                        y=np.concatenate([percentiles[4], percentiles[1][::-1]]),
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(color='rgba(255, 0, 0, 0)'),
                        showlegend=False,
                        name='Total - 5-95th Percentile'
                    ),
                    row=4, col=1
                )

                # 25-75th percentile area
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_vals, x_vals[::-1]]),
                        y=np.concatenate([percentiles[3], percentiles[2][::-1]]),
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.4)',
                        line=dict(color='rgba(255, 0, 0, 0)'),
                        showlegend=False,
                        name='Total - 25-75th Percentile'
                    ),
                    row=4, col=1
                )

                # Mean line
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=mean_forecast,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Total - Mean Forecast'
                    ),
                    row=4, col=1
                )




            
            else:
                # Red forecast lines for Total
                for i in range(self.number_of_paths):
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=self.variables_forecast[i, index_to_display, :],
                            mode='lines',
                            line=dict(color='red', width=2),
                            opacity=0.2,
                            showlegend=False
                        ),
                        row=4, col=1
                    )

            title_text=f'Variable: {variable}, ({index_to_display})'

            if title_addition is not None:
                title_text+= title_addition

            fig.update_layout(
                height=1000,
                width=1200,
                title_text=title_text,
                showlegend=True
            )

        #### THIS IS THE ELSE IF WE DON'T HAVE THE FD

        else:
            
            title_text=f'Variable: {variable}, ({index_to_display})'
            
            fig = go.Figure()

            len_x = len(self.df_initial.iloc[index_to_display, 2:].to_numpy())
            x_vals = np.arange(len_x, len_x+self.variables_forecast.shape[2])

            black_x = np.arange(1,len_x+1)
            blue_x = np.arange(len_x-len(self.df_initial.iloc[index_to_display, 2:].to_numpy()), len_x)
            
            # Blue line for Total
            fig.add_trace(
                go.Scatter(
                    x= blue_x,
                    y=self.df_initial.iloc[index_to_display, 2:].to_numpy(),
                    mode='lines',
                    line=dict(color='blue', width=2),
                    opacity=0.5,
                    name='Total - Initial'
                ),
            )
            # Black line for Total
            fig.add_trace(
                go.Scatter(
                    x=black_x,
                    y=self.variables_df_time_series.iloc[index_to_display, 2:].to_numpy(),
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Total - Variables'
                ),
            )

            if distribution:
                forecasts = self.variables_forecast[:, index_to_display, :]
                mean_forecast = np.mean(forecasts, axis=0)
                percentiles = np.percentile(forecasts, [0, 5, 25, 75, 95, 100], axis=0)

                # x_vals = self.processes["Monthly cicle"].interpolated_forecasts_timeline+len_x

                # 0-100th percentile area
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_vals, x_vals[::-1]]),
                        y=np.concatenate([percentiles[5], percentiles[0][::-1]]),
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.1)',
                        line=dict(color='rgba(255, 0, 0, 0)'),
                        showlegend=False,
                        name='Total - 0-100th Percentile'
                    ),
                )

                # 5-95th percentile area
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_vals, x_vals[::-1]]),
                        y=np.concatenate([percentiles[4], percentiles[1][::-1]]),
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(color='rgba(255, 0, 0, 0)'),
                        showlegend=False,
                        name='Total - 5-95th Percentile'
                    ),
                )

                # 25-75th percentile area
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_vals, x_vals[::-1]]),
                        y=np.concatenate([percentiles[3], percentiles[2][::-1]]),
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.4)',
                        line=dict(color='rgba(255, 0, 0, 0)'),
                        showlegend=False,
                        name='Total - 25-75th Percentile'
                    ),
                )

                # Mean line
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=mean_forecast,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Total - Mean Forecast'
                    ),
                )
            else:
                # Red forecast lines for Total
                for i in range(self.number_of_paths):
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=self.variables_forecast[i, index_to_display, :],
                            mode='lines',
                            line=dict(color='red', width=2),
                            opacity=0.2,
                            showlegend=False
                        ),
                    )

            title_text=f'Variable: {variable}, ({index_to_display})'

            if title_addition is not None:
                title_text+= title_addition

            fig.update_layout(
                title_text=title_text,
                showlegend=True
            )



 
        fig.show()








def log_index_to_log_returns(df: pd.DataFrame):

    nondates=3

    dates = list(df.index)[nondates:]

    new_df = df.copy()

    if df["Datashape"]=="LogIndex":
        jump= jump_dict[df["frequency"]]

        new_df[dates[jump:]]=df[dates[jump:]].to_numpy()-df[dates[:-jump]].to_numpy()

        new_df[dates[:jump]]= np.NaN

        new_df["Datashape"]="LogReturn_F"



    return new_df

def log_returns_to_log_index_1(df: pd.DataFrame):

    
    dates = df.columns.to_list()
    dates.remove("Asset")
    dates.remove("Datashape")

    new_df=df

    new_df.loc[new_df["Datashape"]=="LogReturn",dates]=df.loc[df["Datashape"]=="LogReturn",dates].cumsum(axis=1)

    new_df.loc[new_df["Datashape"]=="LogReturn","Datashape"]="LogIndex"



    return new_df
    

        