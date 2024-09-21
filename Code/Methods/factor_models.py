from .utils.utils_methods import*
from .utils.priors import *
from tqdm import tqdm
import plotly.graph_objects as go
from scipy.stats import jarque_bera, ks_1samp, norm, chi2, uniform

class VAR_process:
    def __init__(self, number_of_lags: int , parameters: list[np.ndarray], phi_0: np.ndarray, time_series: np.ndarray, covariance: np.ndarray, jumps:int, diagonal:bool=False)->None:
        self.dim=time_series.shape[0]
        self.number_of_lags=number_of_lags
        self.parameters=parameters
        self.phi_0=phi_0
        self.time_series=time_series
        self.forecasts=None
        self.covariance=covariance
        self.date=None
        self.diagonal=diagonal
        self.jumps=jumps

        if time_series is not None:
            self.date=time_series.shape[1]
    
    def forecast(self, time_steps, number_of_paths:int, jumps:int):
        np.random.seed(1)
        if number_of_paths==0:

            process=np.concatenate([self.time_series, np.zeros([self.dim, time_steps])], axis=1)

            starter=self.time_series.shape[1]

            for i in range(starter, starter+time_steps):
                process[:,i]=np.random.multivariate_normal(np.zeros(self.dim), self.covariance)+self.phi_0

                for p in range(self.number_of_lags):
                    process[:,i]=process[:,i].T+self.parameters[p]@process[:,i-p*jumps-1].T


            self.forecasts=process[:,-time_steps-1:]

            return process
        
        elif self.diagonal:
            
            self.forecasts=[]

            self.diagonal_parameters = [np.diag(param) for param in self.parameters]
            self.std_diag = np.sqrt(np.diag(self.covariance))

            for _ in range(number_of_paths):

                process=np.concatenate([self.time_series, np.zeros([self.dim, time_steps])], axis=1)

                starter=self.time_series.shape[1]

                for i in range(starter, starter+time_steps):
                    process[:,i]=self.phi_0+self.std_diag*np.random.standard_normal(self.dim)

                    for p in range(self.number_of_lags):
                        process[:,i]=process[:,i].T+self.diagonal_parameters[p] * process[:,i-p*jumps-1].T



                self.forecasts.append(process[:,-time_steps-1:].squeeze())

            self.forecasts=np.array(self.forecasts)

            if self.date is None:
                self.forecasts_timeline=None
            else:
                self.forecasts_timeline=np.linspace(self.date, self.date+jumps*time_steps, time_steps+1)

        else:
            self.forecasts=[]
            for _ in range(number_of_paths):
                process=np.concatenate([self.time_series, np.zeros([self.dim, time_steps])], axis=1)

                starter=self.time_series.shape[1]

                for i in range(starter, starter+time_steps):
                    process[:,i]=np.random.multivariate_normal(np.zeros(self.dim), self.covariance)+self.phi_0

                    for p in range(self.number_of_lags):
                        process[:,i]=process[:,i].T+self.parameters[p]@process[:,i-p*jumps-1].T



                self.forecasts.append(process[:,-time_steps-1:].squeeze())

            self.forecasts=np.array(self.forecasts)

            if self.date is None:
                self.forecasts_timeline=None
            else:
                self.forecasts_timeline=np.linspace(self.date, self.date+jumps*time_steps, time_steps+1)

    def interpolate_forecasts(self, number_of_points: int, method:str="third"):

        if method=="first":

            if len(self.forecasts.shape)==2:

                self.transform_dict=discrete_fourier_transform(self.forecasts)

                times=self.transform_dict["time_line"]

                new_timeline=np.linspace(times[0], times[-1], number_of_points*(len(times)-1)+1)

                len_V=self.dim
                len_f=len(times)
                len_t=len(new_timeline)
                
                frequencies=self.transform_dict["Fourier frequencies"]
                A=self.transform_dict["A"]
                B=self.transform_dict["B"]

                arg_matrix=np.array(frequencies).reshape([len_f,1])@new_timeline.reshape([1,len_t])

                new_A=np.tile(A.reshape([len_V, len_f, 1]), [1,1,len_t])
                new_B=np.tile(B.reshape([len_V, len_f, 1]), [1,1,len_t])

                cos_matrix=np.tile(np.cos(arg_matrix).reshape([1,len_f, len_t]), [len_V,1,1])
                sin_matrix=np.tile(np.sin(arg_matrix).reshape([1,len_f, len_t]), [len_V,1,1])

                new_timeseries=np.sum(new_A*cos_matrix+new_B*sin_matrix, axis=1)

                self.interpolated_forecast=new_timeseries

                return new_timeseries
            
            else:

                self.interpolated_forecast=[]

                for forecast in self.forecasts:

                    self.transform_dict=discrete_fourier_transform(forecast)

                    times=self.transform_dict["time_line"]

                    new_timeline=np.linspace(times[0], times[-1], number_of_points*(len(times)-1)+1)

                    len_V=self.dim
                    len_f=len(times)
                    len_t=len(new_timeline)
                    
                    frequencies=self.transform_dict["Fourier frequencies"]
                    A=self.transform_dict["A"]
                    B=self.transform_dict["B"]

                    arg_matrix=np.array(frequencies).reshape([len_f,1])@new_timeline.reshape([1,len_t])

                    new_A=np.tile(A.reshape([len_V, len_f, 1]), [1,1,len_t])
                    new_B=np.tile(B.reshape([len_V, len_f, 1]), [1,1,len_t])

                    cos_matrix=np.tile(np.cos(arg_matrix).reshape([1,len_f, len_t]), [len_V,1,1])
                    sin_matrix=np.tile(np.sin(arg_matrix).reshape([1,len_f, len_t]), [len_V,1,1])

                    new_timeseries=np.sum(new_A*cos_matrix+new_B*sin_matrix, axis=1)

                    self.interpolated_forecast.append(new_timeseries)

                self.interpolated_forecast=np.array(self.interpolated_forecast)

                if self.date is None:
                    self.interpolated_forecasts_timeline=None
                else:
                    self.interpolated_forecasts_timeline=np.arange(self.date, self.forecasts_timeline[-1]+1)

        elif method=="second":

            base_element = self.time_series[:,self.time_series.shape[1]%number_of_points: -1 : number_of_points]

            base_element = np.nan_to_num(base_element)

            points_to_interpolate_on = np.concatenate([np.tile(base_element.reshape([1, base_element.shape[0], base_element.shape[1]]), [self.forecasts.shape[0],1,1]), self.forecasts], axis=2)

            to_keep = (self.forecasts.shape[2]-1)*number_of_points+1
            if number_of_points==1:
                interpolated_whole_timeseries = points_to_interpolate_on
            else:
                interpolated_whole_timeseries = discrete_fourier_transform_2(points_to_interpolate_on, number_of_points)

            # plt.plot(interpolated_whole_timeseries[0,0,:])
            # plt.plot(self.time_series[0,self.time_series.shape[1]%number_of_points:])
            # plt.show()

            self.interpolated_forecast=interpolated_whole_timeseries[:,:, -to_keep:]

            if self.date is None:
                self.interpolated_forecasts_timeline=None
            else:
                self.interpolated_forecasts_timeline=np.arange(self.date, self.forecasts_timeline[-1]+1)

        elif method=="third":

           
            if number_of_points==1:
                self.interpolated_forecast = self.forecasts
            else:
                base_element = self.time_series[:,self.time_series.shape[1]%number_of_points: -1 : number_of_points]

                base_element = np.nan_to_num(base_element)

                points_to_interpolate_on = np.concatenate([np.tile(base_element.reshape([1, base_element.shape[0], base_element.shape[1]]), [self.forecasts.shape[0],1,1]), self.forecasts], axis=2)

                to_keep = (self.forecasts.shape[2]-2)*number_of_points+1
                
                interpolated_whole_timeseries = discrete_fourier_transform_2(points_to_interpolate_on, number_of_points)

            
                t = np.array([0,1,number_of_points+1,number_of_points+2])
                t_interp = np.arange(1, number_of_points+1)
                f = np.concatenate([np.tile(self.time_series[:, -2:].reshape([1,self.time_series.shape[0],2]), [interpolated_whole_timeseries.shape[0],1,1]), interpolated_whole_timeseries[:,:, -to_keep:-to_keep+2]], axis=2)

                x_interp = multisplines(t,f,t_interp)


                self.interpolated_forecast=np.concatenate([x_interp, interpolated_whole_timeseries[:,:, -to_keep:]], axis= 2)

            if self.date is None:
                self.interpolated_forecasts_timeline=None
            else:
                self.interpolated_forecasts_timeline=np.arange(self.date, self.forecasts_timeline[-1]+1)

        else:
            raise TypeError(f"Unknown method: {method}.")
        
    def get_log_likelihood(self):

        beta = vectorize_parameters(self.phi_0, self.parameters)[0]

        X_0 = data_to_X_0(self.time_series, self.number_of_lags, self.jumps)

        X_1 = data_to_X_1(self.time_series, self.number_of_lags, self.jumps)

        time_size = X_0.shape[0]

        U     =   X_1-(kron_identity(self.dim, X_0) @ beta.reshape([len(beta),1]))
        
        sigma_inv = np.linalg.inv(self.covariance)

        Xi_inv = identity_kron(time_size, sigma_inv)
                           
        corp= U.T @ Xi_inv @ U

        det = np.linalg.det(self.covariance)

        log_likelihood= -0.5* corp -(self.dim*time_size/2)*np.log((2*np.pi))-0.5*np.log(det )*time_size

        return log_likelihood.squeeze()
    
    def get_information_criterias(self):

        log_likelihood = self.get_log_likelihood()


        K, T = self.time_series.shape
        p = self.number_of_lags
        j = self.jumps

        num_param = K * (K * (p +1) + 1)
        

        AIC = -2* log_likelihood + 2 * num_param
        BIC = -2* log_likelihood + num_param * np.log(T)
        HQC = -2* log_likelihood + 2 *num_param * np.log(np.log(T))

        return {
            "Log-Likelihood":log_likelihood,
            "AIC": AIC,
            "BIC": BIC,
            "HQC": HQC
        }

    def get_error_tests(self):
        
        X_0 = data_to_X_0(self.time_series, self.number_of_lags, self.jumps)
        X_2 = data_to_X_2(self.time_series, self.number_of_lags, self.jumps)

        time_size = X_0.shape[0]
        K_p = X_0.shape[1]
        K= X_2.shape[0]

        beta = vectorize_parameters(self.phi_0, self.parameters)[0]
        B = beta.reshape([K, K_p], order="F")
        V     =   X_2-(B @ X_0.T)

        Sigma = self.covariance

        assert V.shape == (K, time_size)

        # FIRST let's check the expectation

        mean = np.mean(V, axis=1, keepdims=True)

        Sigma_hat = (V-mean) @ (V-mean).T / (time_size-1)

        mean_max_error = np.max(np.abs(mean))
        mean_CLT_diff = mean_max_error*time_size

        cov_max_error = np.max(np.abs(Sigma_hat-Sigma))

        # JB normality test

        V = np.flip(np.flip(V, axis=1)[:,::self.jumps], axis=1)

        p_values = np.array([jarque_bera(JB_data)[1] for JB_data in V])

        alpha = 0.05

        counted = np.sum(np.where(p_values < alpha, 1, 0))

        JB_fails = counted/K

        max_eigen = var_stability_test(B)



        ### MARDIA's test
        
        A_stat_adj, B_stat_adj, A_p_value_adj, B_p_value_adj, df_A_adj = mardia_test(V, Sigma=self.covariance, zero_mean=True)
        A_stat, B_stat, A_p_value, B_p_value, df_A = mardia_test(V)

        M_stat, M_p_value = multivariate_portmanteau_test(V, self.number_of_lags)
        LM_stat, LM_p_value = multivariate_lm_test(V, self.number_of_lags)

        if self.jumps==1 and self.number_of_lags==1:
            my_fig = go.Figure()
            
            my_fig.add_traces([
                go.Histogram(   x=V[i,:],
                                opacity=0.7,
                                name=f'{i} in J{self.jumps} and L{self.number_of_lags}') 
                                
                                for i in range(self.dim)
            ])
            my_fig.update_layout(barmode='overlay')
            my_fig.show()

            my_fig = go.Figure()

            for i in range(self.dim):
                for j in range(self.dim):
                    if i!=j:

                        my_fig.add_traces(
                            go.Scatter(   x=V[i,:],
                                    y=V[j,:],
                                            opacity=0.7,
                                            name=f'J{self.jumps} and L{self.number_of_lags}',mode='markers'),
                                            
                        )

            my_fig.update_layout(barmode='overlay')
            my_fig.show()

        

        ### DURBIN-WATSON d-test:

        return {
            "mean_max_error": mean_max_error , 
            "mean_CLT_diff": mean_CLT_diff ,
            "cov_max_error": cov_max_error,
            "JB_fails": JB_fails,
            "JB_mean": np.mean(p_values),
            "JB_median": np.median(p_values),
            "A":A_stat,
            "B":B_stat,
            "A_p_value": A_p_value,
            "B_p_value": B_p_value,
            "df_A":df_A,
            "A_adj":A_stat_adj,
            "B_adj":B_stat_adj,
            "A_p_value_adj": A_p_value_adj,
            "B_p_value_adj": B_p_value_adj,
            "df_A_adj":df_A_adj,
            "Max Eigenvalue": max_eigen,
            "M":M_stat,
            "LM":LM_stat,
            "M_p_value": M_p_value,
            "LM_p_value": LM_p_value,
        }






class BVAR_process(VAR_process):
    
    def forecast(self, time_steps, number_of_paths:int, jumps:int):
        np.random.seed(1)
        if number_of_paths==0:

            warnings.warn("Zero forecast asked")

            process=np.concatenate([self.time_series, np.zeros([self.dim, time_steps])], axis=1)

            starter=self.time_series.shape[1]

            for i in range(starter, starter+time_steps):
                process[:,i]=np.random.multivariate_normal(np.zeros(self.dim), self.covariance)+self.phi_0

                for p in range(self.number_of_lags):
                    process[:,i]=process[:,i].T+self.parameters[p]@process[:,i-p*jumps-1].T


            self.forecasts=process[:,-time_steps-1:]

            return process

        else:
            self.forecasts=[]

            starter=self.time_series.shape[1]

            for path in range(number_of_paths):
                process=np.concatenate([self.time_series, np.zeros([self.dim, time_steps])], axis=1)

                for i in range(starter, starter+time_steps):
                    process[:,i]=np.random.multivariate_normal(np.zeros(self.dim), self.covariance[path])+self.phi_0[path]

                    for p in range(self.number_of_lags):
                        process[:,i]=process[:,i].T+self.parameters[path][p]@process[:,i-p*jumps-1].T



                self.forecasts.append(process[:,-time_steps-1:].squeeze())

            self.forecasts=np.array(self.forecasts)

            if self.date is None:
                self.forecasts_timeline=None
            else:
                self.forecasts_timeline=np.linspace(self.date, self.date+jumps*time_steps, time_steps+1)
        
    def get_log_likelihood(self, number_of_paths:int=2000):
        
        X_0 = data_to_X_0(self.time_series, self.number_of_lags, self.jumps)

        X_2 = data_to_X_2(self.time_series, self.number_of_lags, self.jumps)

        time_size = X_0.shape[0]

        K = X_0.shape[1]

        log_likelihoods=[]

        for path in tqdm(range(number_of_paths)):     

            beta = vectorize_parameters(self.phi_0[path], self.parameters[path])[0]

            B = beta.reshape([self.dim, K], order="F")

            V     =   X_2-(B @ X_0.T)

            V = np.flip(np.flip(V, axis=1)[:,::self.jumps], axis=1)
            
            sigma_inv = np.linalg.inv(self.covariance[path])
                            
            corp= np.trace(V.T @ sigma_inv @ V)

            log_likelihood= corp + time_size*np.log(np.linalg.det(self.covariance[path]))

            log_likelihoods.append(log_likelihood)

        log_likelihood_est= -0.5*np.mean(log_likelihoods)-(self.dim*time_size/2)*np.log((2*np.pi))
        log_likelihood_est_std = 0.5*np.std(log_likelihoods)/np.sqrt(number_of_paths)

        return log_likelihood_est, log_likelihood_est_std
    
    def get_information_criterias(self, number_of_paths:int=2000):

        log_likelihood, log_likelihood_std = self.get_log_likelihood(number_of_paths=number_of_paths)

        K, T = self.time_series.shape
        p = self.number_of_lags

        num_param = K * (K*p +1)
        

        AIC = -2* log_likelihood + 2 * num_param
        BIC = -2* log_likelihood + num_param * np.log(T)
        HQC = -2* log_likelihood + 2 *num_param * np.log(np.log(T))

        return {
            "Log-Likelihood":log_likelihood,
            "Log-Likelihood std": log_likelihood_std,
            "AIC": AIC,
            "BIC": BIC,
            "HQC": HQC
        }
    
    def get_error_tests(self, number_of_paths:int=2000):
        
        X_0 = data_to_X_0(self.time_series, self.number_of_lags, self.jumps)
        X_2 = data_to_X_2(self.time_series, self.number_of_lags, self.jumps)

        time_size = X_0.shape[0]
        K_p = X_0.shape[1]
        K= X_2.shape[0]

        mean_max_errors=[]
        cov_max_errors=[]
        max_eigenvals=[]

        # A_stat_list=[]
        # B_stat_list=[]
        # A_p_value_list=[]
        # B_p_value_list=[]
        # df_A_list=[]
        # A_stat_adj_list=[]
        # B_stat_adj_list=[]
        # A_p_value_adj_list=[]
        # B_p_value_adj_list=[]
        # df_A_adj_list=[]
        # M_stat_list=[]
        # LM_stat_list=[]
        # M_p_value_list=[]
        # LM_p_value_list=[]

        for path in tqdm(range(number_of_paths)):

            beta = vectorize_parameters(self.phi_0[path], self.parameters[path])[0]
            B = beta.reshape([K, K_p], order="F")
            V     =   X_2-(B @ X_0.T)

            Sigma = self.covariance[path]

            assert V.shape == (K, time_size)

            # FIRST let's check the expectation

            mean = np.mean(V, axis=1, keepdims=True)

            Sigma_hat = (V-mean) @ (V-mean).T / (time_size-1)

            mean_max_errors.append(np.max(np.abs(mean)))

            cov_max_errors.append(np.max(np.abs(Sigma_hat-Sigma)))

            # V = np.flip(np.flip(V, axis=1)[:,::self.jumps], axis=1)


            max_eigenvals.append(var_stability_test(B))

            ### MARDIA's test
        
            # A_stat_adj, B_stat_adj, A_p_value_adj, B_p_value_adj, df_A_adj = mardia_test(V, Sigma=Sigma, zero_mean=True)
            # A_stat, B_stat, A_p_value, B_p_value, df_A = mardia_test(V)

            # M_stat, M_p_value = multivariate_portmanteau_test(V, self.number_of_lags)
            # LM_stat, LM_p_value = multivariate_lm_test(V, self.number_of_lags)


            # A_stat_list.append(A_stat)
            # B_stat_list.append(B_stat)
            # A_p_value_list.append(A_p_value)
            # B_p_value_list.append(B_p_value)
            # df_A_list.append(df_A)
            # A_stat_adj_list.append(A_stat_adj)
            # B_stat_adj_list.append(B_stat_adj)
            # A_p_value_adj_list.append(A_p_value_adj)
            # B_p_value_adj_list.append(B_p_value_adj)
            # df_A_adj_list.append(df_A_adj)
            # M_stat_list.append(M_stat)
            # LM_stat_list.append(LM_stat)
            # M_p_value_list.append(M_p_value)
            # LM_p_value_list.append(LM_p_value)

        # assert df_A_adj_list[:-1]==df_A_adj_list[1:]

        # def chi2_A(x): return chi2.cdf(x, df = df_A)
        # def chi2_M(x): return chi2.cdf(x, df = K**2 * self.number_of_lags)
        # def chi2_LM(x): return chi2.cdf(x, df = K**2)
        # def stand_norm(x): return norm.cdf(x, loc=0, scale=1)
        # def uniform_01(x): return uniform.cdf(x, loc=0, scale=1)

        # A_stat_test = ks_1samp(A_stat_list, chi2_A)
        # B_stat_test = ks_1samp(B_stat_list, stand_norm)
        # A_stat_adj_test = ks_1samp(A_stat_adj_list, chi2_A)
        # B_stat_adj_test = ks_1samp(B_stat_adj_list, stand_norm)
        # M_stat_test = ks_1samp(M_stat_list, chi2_M)
        # LM_stat_test = ks_1samp(LM_stat_list, chi2_LM)


        # A_p_value_test = ks_1samp(A_p_value_list, uniform_01)
        # B_p_value_test = ks_1samp(B_p_value_list, uniform_01)
        # A_p_value_adj_test = ks_1samp(A_p_value_adj_list, uniform_01)
        # B_p_value_adj_test = ks_1samp(A_p_value_adj_list, uniform_01)
        # M_p_value_test = ks_1samp(M_p_value_list, uniform_01)
        # LM_p_value_test = ks_1samp(LM_p_value_list, uniform_01)

        return {
            # "mean_max_errors_mean"          :        np.mean(mean_max_errors),
            # "mean_max_errors_max"           :        np.max(mean_max_errors),
            # "cov_max_errors_mean"           :        np.mean(cov_max_errors),
            # "cov_max_errors_max"            :        np.max(cov_max_errors),
            "max_eigenvals_mean"            :        np.mean(max_eigenvals),
            "max_eigenvals_max"         :        np.max(max_eigenvals),
            "max_eigenvals_std"         :       np.std(max_eigenvals),
            "max_eigenvals_std_CLT"         :       np.std(max_eigenvals)/np.sqrt(2000),
            # "Num_unstable"              :        sum(1 for x in max_eigenvals if x >= 1)/number_of_paths,
            # "A_stat_test_stat"          :        A_stat_test.statistic,
            # "A_stat_test_p_value"           :        A_stat_test.pvalue,
            # "B_stat_test_stat"          :        B_stat_test.statistic,
            # "B_stat_test_p_value"           :        B_stat_test.pvalue,
            # "A_stat_adj_test_stat"          :        A_stat_adj_test.statistic,
            # "A_stat_adj_test_p_value"           :        A_stat_adj_test.pvalue,
            # "B_stat_adj_test_stat"          :        B_stat_adj_test.statistic,
            # "B_stat_adj_test_p_value"           :        B_stat_adj_test.pvalue,
            # "M_stat_test_stat"          :        M_stat_test.statistic,
            # "M_stat_test_p_value"           :        M_stat_test.pvalue,
            # "LM_stat_test_stat"         :        LM_stat_test.statistic,
            # "LM_stat_test_p_value"          :        LM_stat_test.pvalue,
            # "A_p_value_test_stat"           :        A_p_value_test.statistic,
            # "A_p_value_test_p_value"            :        A_p_value_test.pvalue,
            # "B_p_value_test_stat"           :        B_p_value_test.statistic,
            # "B_p_value_test_p_value"            :        B_p_value_test.pvalue,
            # "A_p_value_adj_test_stat"           :        A_p_value_adj_test.statistic,
            # "A_p_value_adj_test_p_value"            :        A_p_value_adj_test.pvalue,
            # "B_p_value_adj_test_stat"           :        B_p_value_adj_test.statistic,
            # "B_p_value_adj_test_p_value"            :        B_p_value_adj_test.pvalue,
            # "M_p_value_test_stat"           :        M_p_value_test.statistic,
            # "M_p_value_test_p_value"            :        M_p_value_test.pvalue,
            # "LM_p_value_test_stat"          :        LM_p_value_test.statistic,
            # "LM_p_value_test_p_value"           :        LM_p_value_test.pvalue,
            # "A_p_value_max":                np.max(A_p_value_list),
            # "A_p_value_min":                np.min(A_p_value_list),
            # "B_p_value_max":                np.max(B_p_value_list),
            # "B_p_value_min":                np.min(B_p_value_list),
            # "A_p_value_adj_max":                np.max(A_p_value_adj_list),
            # "A_p_value_adj_min":                np.min(A_p_value_adj_list),
            # "B_p_value_adj_max":                np.max(B_p_value_adj_list),
            # "B_p_value_adj_min":                np.min(B_p_value_adj_list),
            # "M_p_value_max":                np.max(M_p_value_list),
            # "M_p_value_min":                np.min(M_p_value_list),
            # "LM_p_value_max":               np.max(LM_p_value_list),
            # "LM_p_value_min":               np.min(LM_p_value_list),
            }


'''
class BVAR_process_1:

    def __init__(self, dim=None, number_of_lags=None, prior: prior= prior(), time_series=None, jumps:int=1) -> None:
        
        self.dim=dim
        self.number_of_lags=number_of_lags
        self.prior = prior
        self.time_series=time_series
        self.forecasts=None
        self.date=None

        if time_series is not None:
            self.date=time_series.shape[1]

        self.prior.aggregate_likelihood(data=time_series, number_of_lags=self.number_of_lags, jumps=jumps)

    def forecast(self, time_steps,  number_of_paths:int=1, jumps:int=1):

        samples= self.prior.sample(number_of_paths)
        
        self.forecasts=[]
        
        starter=self.time_series.shape[1]

        for n_p in range(number_of_paths):
            process=np.concatenate([self.time_series, np.zeros([self.dim, time_steps])], axis=1)

            beta = samples["beta"][n_p]
            Sigma = samples["Sigma"][n_p]

            phi_0, parameters= beta_to_parameters(beta=beta, dim= self.dim, number_of_lags=self.number_of_lags)

            for i in range(starter, starter+time_steps):

                

                

                process[:,i]=np.random.multivariate_normal(np.zeros(self.dim), Sigma)+phi_0


                for p in range(self.number_of_lags):
                    process[:,i]=process[:,i].T+parameters[p]@process[:,i-p*jumps-1].T



            self.forecasts.append(process[:,-time_steps-1:].squeeze())

        self.forecasts=np.array(self.forecasts)

        if self.date is None:
            self.forecasts_timeline=None
        else:
            self.forecasts_timeline=np.linspace(self.date, self.date+jumps*time_steps, time_steps+1)


    def interpolate_forecasts(self, number_of_points: int, method:str="second"):

        if method=="first":

            if len(self.forecasts.shape)==2:

                self.transform_dict=discrete_fourier_transform(self.forecasts)

                times=self.transform_dict["time_line"]

                new_timeline=np.linspace(times[0], times[-1], number_of_points*(len(times)-1)+1)

                len_V=self.dim
                len_f=len(times)
                len_t=len(new_timeline)
                
                frequencies=self.transform_dict["Fourier frequencies"]
                A=self.transform_dict["A"]
                B=self.transform_dict["B"]

                arg_matrix=np.array(frequencies).reshape([len_f,1])@new_timeline.reshape([1,len_t])

                new_A=np.tile(A.reshape([len_V, len_f, 1]), [1,1,len_t])
                new_B=np.tile(B.reshape([len_V, len_f, 1]), [1,1,len_t])

                cos_matrix=np.tile(np.cos(arg_matrix).reshape([1,len_f, len_t]), [len_V,1,1])
                sin_matrix=np.tile(np.sin(arg_matrix).reshape([1,len_f, len_t]), [len_V,1,1])

                new_timeseries=np.sum(new_A*cos_matrix+new_B*sin_matrix, axis=1)

                self.interpolated_forecast=new_timeseries

                return new_timeseries
            
            else:

                self.interpolated_forecast=[]

                for forecast in self.forecasts:

                    self.transform_dict=discrete_fourier_transform(forecast)

                    times=self.transform_dict["time_line"]

                    new_timeline=np.linspace(times[0], times[-1], number_of_points*(len(times)-1)+1)

                    len_V=self.dim
                    len_f=len(times)
                    len_t=len(new_timeline)
                    
                    frequencies=self.transform_dict["Fourier frequencies"]
                    A=self.transform_dict["A"]
                    B=self.transform_dict["B"]

                    arg_matrix=np.array(frequencies).reshape([len_f,1])@new_timeline.reshape([1,len_t])

                    new_A=np.tile(A.reshape([len_V, len_f, 1]), [1,1,len_t])
                    new_B=np.tile(B.reshape([len_V, len_f, 1]), [1,1,len_t])

                    cos_matrix=np.tile(np.cos(arg_matrix).reshape([1,len_f, len_t]), [len_V,1,1])
                    sin_matrix=np.tile(np.sin(arg_matrix).reshape([1,len_f, len_t]), [len_V,1,1])

                    new_timeseries=np.sum(new_A*cos_matrix+new_B*sin_matrix, axis=1)

                    self.interpolated_forecast.append(new_timeseries)

                self.interpolated_forecast=np.array(self.interpolated_forecast)

                if self.date is None:
                    self.interpolated_forecasts_timeline=None
                else:
                    self.interpolated_forecasts_timeline=np.arange(self.date, self.forecasts_timeline[-1]+1)

        elif method=="second":

            self.interpolated_forecast=discrete_fourier_transform_2(self.forecasts, number_of_points)

            if self.date is None:
                self.interpolated_forecasts_timeline=None
            else:
                self.interpolated_forecasts_timeline=np.arange(self.date, self.forecasts_timeline[-1]+1)

        else:
            raise TypeError(f"Unknown method: {method}.")
    '''