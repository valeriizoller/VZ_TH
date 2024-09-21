import numpy as np
from .distributions import *
from.utils_methods import *   
from .truncated_mvn_sampler import *

class Minnesota_Prior:
    def __init__(self, dim:int, number_of_lags:int, lambda_1, lambda_2, lambda_3, high_variance_for_phi_0=np.inf) -> None:

        N=dim*(number_of_lags*dim+1)
        beta_0=np.zeros(N)

        for j in range(1,dim+1):
            beta_0[j*(dim+1)-1]=1

        # Here we deal with the variances yeah!

        phi_0_var=np.ones([dim,1])*high_variance_for_phi_0

        parameters_var=[]

        for i in range(number_of_lags):
            Phi_i_var=np.zeros([dim,dim])
            comm=lambda_1**2/((i+1)**lambda_3)
            for a in range(dim):
                for b in range(dim):
                    if a==b:
                        Phi_i_var[a,b]=comm
                    else:
                        Phi_i_var[a,b]=lambda_2**2*comm
            parameters_var.append(Phi_i_var)

        Omega_0=np.diag(vectorize_parameters(phi_0_var, parameters_var)[0].squeeze())

        self.prior_hyperparameters= {  "beta_0": beta_0.squeeze(),
                                        "Omega_0": Omega_0}


    def aggregate_likelihood(self, data, number_of_lags:int=1, jumps:int=1):

        OLS_estimator = OLS_new_estimator(data, number_of_lags=number_of_lags, jumps=jumps)

        mu = OLS_estimator["beta_OLS"]
        Sigma = OLS_estimator["Sigma"]

        # Sigma = strongly_diagonalize(Sigma)

        Lambda = Sigma_to_Lambda(Sigma, data, number_of_lags, jumps)


        beta_0 = self.prior_hyperparameters["beta_0"]
        Omega_0 = self.prior_hyperparameters["Omega_0"]

        Lambda_inv=np.linalg.inv(Lambda)
        Omega_0_inv = np.linalg.inv(Omega_0)

        Omega_1 = np.linalg.inv(Lambda_inv+Omega_0_inv)
        beta_1 = Omega_1 @ (Lambda_inv @ mu + Omega_0_inv @ beta_0.reshape([len(beta_0),1])).reshape([len(beta_0),1])

        self.posterior_hyperparameters={    "beta_1": beta_1.squeeze(),
                                            "Omega_1": Omega_1,
                                            "Sigma": Sigma   }

            

    def sample(self, size)->dict[str, np.ndarray[np.float_]]:
        betas=[]
        

        beta_1 = self.posterior_hyperparameters["beta_1"]
        Omega_1 = self.posterior_hyperparameters["Omega_1"]
        Sigma = self.posterior_hyperparameters["Sigma"]

        Sigmas=[Sigma for _ in range(size)]

        for _ in tqdm(range(size)):
            betas.append(np.random.multivariate_normal(beta_1, Omega_1))

        sample_dict={"betas": betas,
                     "Sigmas": Sigmas}
        
        return sample_dict

class Uniform_prior:
    def __init__(self, interval=np.ndarray[np.float_]) -> None:

        self.prior_hyperparameters= {"beta_Intervals": interval}


    def aggregate_likelihood(self, data, number_of_lags:int=1, jumps:int=1):

        OLS_estimator = OLS_new_estimator(data, number_of_lags=number_of_lags, jumps=jumps)

        mu = OLS_estimator["beta_OLS"]
        Sigma = OLS_estimator["Sigma"]

        Lambda = Sigma_to_Lambda(Sigma, data, number_of_lags, jumps)

        Omega_1 = Lambda
        beta_1 = mu

        self.posterior_hyperparameters={    "beta_Mean": beta_1.squeeze(),
                                            "beta_Covariance_Matrix": Omega_1,
                                            "Sigma": Sigma,
                                            "Intervals": self.prior_hyperparameters["beta_Intervals"]}
        
        self.posterior_distribution = TruncatedMVN(mu=self.posterior_hyperparameters["beta_Mean"], 
                                                   cov= self.posterior_hyperparameters["beta_Covariance_Matrix"], 
                                                   lb=self.posterior_hyperparameters["Intervals"][:,0], 
                                                   ub=self.posterior_hyperparameters["Intervals"][:,1])

    def sample(self, size)->dict[str, np.ndarray[np.float_]]:
        betas=[]
        Sigmas=[self.posterior_hyperparameters["Sigma"] for _ in range(size)]

        sampled_beta = self.posterior_distribution.sample(size)

        for i in range(size):
            betas.append(sampled_beta[:,i].squeeze())       

        sample_dict={"betas": betas,
                     "Sigmas": Sigmas}
        
        return sample_dict

class Independent_Normal_inv_Wishart:
    def __init__(self, beta_0, Omega_0, S_0, nu_0):

        self.prior_hyperparameters= {"beta_0": beta_0.squeeze(),
                                    "Omega_0": Omega_0,
                                    "S_0": S_0,
                                    "nu_0": nu_0}
        
    def aggregate_likelihood(self, data, number_of_lags:int=1, jumps:int=1):

        beta_0 = self.prior_hyperparameters["beta_0"]
        Omega_0 = self.prior_hyperparameters["Omega_0"]
        nu_0 =self.prior_hyperparameters['nu_0']
        S_0 = self.prior_hyperparameters['S_0']

        OLS_estimator = OLS_new_estimator(data, number_of_lags=number_of_lags, jumps=jumps)

        mu = OLS_estimator["beta_OLS"]
        Sigma = OLS_estimator["Sigma"]

        Lambda = Sigma_to_Lambda(Sigma, data, number_of_lags, jumps)

        # NEW CHANGE

        Omega_0 = np.diag(np.diag(Lambda))

        Lambda_inv=np.linalg.inv(Lambda)
        Omega_0_inv = np.linalg.inv(Omega_0)

        Omega_1 = np.linalg.inv(Lambda_inv+Omega_0_inv)
        beta_1 = Omega_1 @ (Lambda_inv @ mu + Omega_0_inv @ beta_0.reshape([len(beta_0),1])).reshape([len(beta_0),1])

        X_0 = data_to_X_0(data, number_of_lags, jumps)
        X_2 = data_to_X_2(data, number_of_lags, jumps)

        T = data.shape[1]
        p= number_of_lags*jumps

        nu_1= nu_0+T-p

        self.posterior_hyperparameters={"beta_0": beta_0.squeeze(),
                                        "Omega_0": Omega_0,
                                        "S_0": S_0,
                                        "nu_1": nu_1,
                                        "X_0": X_0,
                                        "X_2": X_2,
                                        "mu": mu,
                                        "starting_beta": beta_1,
                                        "T":T,
                                        "p":p}
        
    def sample(self, size, gibbs_burnin=0)->dict[str, np.ndarray[np.float_]]:

        betas=[]
        Sigmas=[]

        beta = self.posterior_hyperparameters["starting_beta"]
        nu_1 = self.posterior_hyperparameters["nu_1"]
        X_2 =self.posterior_hyperparameters["X_2"]
        X_0 =self.posterior_hyperparameters["X_0"]
        mu =self.posterior_hyperparameters["mu"]
        S_0 =self.posterior_hyperparameters["S_0"]
        Omega_0 = self.posterior_hyperparameters["Omega_0"]
        beta_0 = self.posterior_hyperparameters["beta_0"]
        T = self.posterior_hyperparameters["T"]
        p = self.posterior_hyperparameters["p"]

        Omega_0_inv = np.linalg.inv(Omega_0)

        for i in range(size+gibbs_burnin):

            if i==gibbs_burnin:
                print("Burn in finshed!")
            # Sample Sigma
            S_L = get_S_L_from_beta(beta, X_0, X_2)
            S_1= S_0 + S_L

            S_1_inv = np.linalg.inv(S_1)

            Sigma_inv = sample_wishart(S_1_inv, nu_1, 1).squeeze()

            Xi_inv = identity_kron(T-p, Sigma_inv)

            X_0_kron_I_N = kron_identity(X_2.shape[0], X_0)

            Lambda_inv = X_0_kron_I_N.T @ Xi_inv @ X_0_kron_I_N

            Omega_1 = np.linalg.inv(Lambda_inv+Omega_0_inv)
            beta_1 = Omega_1 @ (Lambda_inv @ mu + Omega_0_inv @ beta_0.reshape([len(beta_0),1])).reshape([len(beta_0),1]).squeeze()

            beta = np.random.multivariate_normal(beta_1, Omega_1)

            Sigma = np.linalg.inv(Sigma_inv)

            betas.append(beta)
            Sigmas.append(Sigma)

        sample_dict={"betas": betas[-size:],
                     "Sigmas": Sigmas[-size:]}
        
        return sample_dict


class Conjugate_Normal_inv_Wishart:
    def __init__(self, beta_0, Psi_0, S_0, nu_0):

        self.prior_hyperparameters= {"beta_0": beta_0.squeeze(),
                                    "Psi_0": Psi_0,
                                    "S_0": S_0,
                                    "nu_0": nu_0}
        
    def aggregate_likelihood(self, data, number_of_lags:int=1, jumps:int=1):

        beta_0 = self.prior_hyperparameters['beta_0']
        Psi_0 = self.prior_hyperparameters['Psi_0']
        S_0 = self.prior_hyperparameters['S_0']
        nu_0 = self.prior_hyperparameters['nu_0']

        X_0 = data_to_X_0(data, number_of_lags, jumps)
        X_2 = data_to_X_2(data, number_of_lags, jumps)

        T = data.shape[1]
        p = number_of_lags * jumps
        N = X_2.shape[0]

        B_0 = beta_0.reshape([N , X_0.shape[1]], order="F")

        X_0_sqr = X_0.T @ X_0

        pseudo_inv = np.linalg.inv(X_0_sqr)

        B_tilde = X_2 @ X_0 @ pseudo_inv

        S_L = get_S_L(data, number_of_lags, jumps)


        S_1 = S_0+S_L

        nu_1= nu_0+T-p+1

        # CHANGE 1
        Psi_0_inv = np.diag(np.diag(X_0_sqr)) #np.linalg.inv(Psi_0)

        Psi_1 = np.linalg.inv(Psi_0_inv + X_0_sqr)
        B_1 = (B_0 @ Psi_0_inv + B_tilde @ X_0_sqr)@ Psi_1

        beta_1 = vectorize_matrix(B_1).squeeze()

        self.posterior_hyperparameters={"beta_1": beta_1,
                                        "Psi_1": Psi_1,
                                        "S_1": S_1,
                                        "nu_1": nu_1}
        
    def sample(self, size)->dict[str, np.ndarray[np.float_]]:

        betas=[]
        Sigmas=[]

        nu_1 = self.posterior_hyperparameters["nu_1"]
        S_1 =self.posterior_hyperparameters["S_1"]
        Psi_1 = self.posterior_hyperparameters["Psi_1"]
        beta_1 = self.posterior_hyperparameters["beta_1"]

        sampled_Sigma = sample_inverse_wishart(S_1, nu_1, size)

        for Sigma in sampled_Sigma:
            Sigmas.append(Sigma)

            Omega_1 = np.kron(Sigma, Psi_1)

            beta = np.random.multivariate_normal(beta_1, Omega_1)

            betas.append(beta)

        sample_dict={"betas": betas,
                     "Sigmas": Sigmas}
        
        return sample_dict
    

class Zellners:
    def __init__(self, beta_0, Omega_0):

        self.prior_hyperparameters= {"beta_0": beta_0.squeeze(),
                                    "Omega_0": Omega_0}
        
    def aggregate_likelihood(self, data, number_of_lags:int=1, jumps:int=1):

        beta_0 = self.prior_hyperparameters["beta_0"]
        Omega_0 = self.prior_hyperparameters["Omega_0"]

        OLS_estimator = OLS_new_estimator(data, number_of_lags=number_of_lags, jumps=jumps)

        mu = OLS_estimator["beta_OLS"]
        Sigma = OLS_estimator["Sigma"]

        Lambda = Sigma_to_Lambda(Sigma, data, number_of_lags, jumps)

        # NEW CHANGE

        Omega_0 = np.diag(np.diag(Lambda))

        Lambda_inv=np.linalg.inv(Lambda)
        Omega_0_inv = np.linalg.inv(Omega_0)

        Omega_1 = np.linalg.inv(Lambda_inv+Omega_0_inv)
        beta_1 = Omega_1 @ (Lambda_inv @ mu + Omega_0_inv @ beta_0.reshape([len(beta_0),1])).reshape([len(beta_0),1])

        X_0 = data_to_X_0(data, number_of_lags, jumps)
        X_2 = data_to_X_2(data, number_of_lags, jumps)

        T = data.shape[1]
        p= number_of_lags*jumps

        nu_1= T-p

        self.posterior_hyperparameters={"beta_0": beta_0.squeeze(),
                                        "Omega_0": Omega_0,
                                        "nu_1": nu_1,
                                        "X_0": X_0,
                                        "X_2": X_2,
                                        "mu": mu,
                                        "starting_beta": beta_1,
                                        "T":T,
                                        "p":p,
                                        "N": data.shape[0]}
        
    def sample(self, size, gibbs_burnin=0)->dict[str, np.ndarray[np.float_]]:

        betas=[]
        Sigmas=[]

        beta = self.posterior_hyperparameters["starting_beta"]
        nu_1 = self.posterior_hyperparameters["nu_1"]
        X_2 =self.posterior_hyperparameters["X_2"]
        X_0 =self.posterior_hyperparameters["X_0"]
        mu =self.posterior_hyperparameters["mu"]
        Omega_0 = self.posterior_hyperparameters["Omega_0"]
        beta_0 = self.posterior_hyperparameters["beta_0"]
        T = self.posterior_hyperparameters["T"]
        p = self.posterior_hyperparameters["p"]
        N = self.posterior_hyperparameters["N"]

        Omega_0_inv = np.linalg.inv(Omega_0)

        X_0_kron_I_K = kron_identity(N, X_0)

        for i in range(size+gibbs_burnin):

            if i==gibbs_burnin:
                print("Burn in finshed!")

            # Sample Sigma
            S_1 = get_S_L_from_beta(beta, X_0, X_2)

            S_1_inv = np.linalg.inv(S_1)

            Sigma_inv = sample_wishart(S_1_inv, nu_1, 1).squeeze()

            Xi_inv = identity_kron(T-p, Sigma_inv)

            X_0_kron_I_N = kron_identity(X_2.shape[0], X_0)

            Lambda_inv = X_0_kron_I_N.T @ Xi_inv @ X_0_kron_I_N

            Omega_1 = np.linalg.inv(Lambda_inv+Omega_0_inv)
            beta_1 = Omega_1 @ (Lambda_inv @ mu + Omega_0_inv @ beta_0.reshape([len(beta_0),1])).reshape([len(beta_0),1]).squeeze()

            beta = np.random.multivariate_normal(beta_1, Omega_1)

            Sigma = np.linalg.inv(Sigma_inv)

            betas.append(beta)
            Sigmas.append(Sigma)

        sample_dict={"betas": betas[-size:],
                     "Sigmas": Sigmas[-size:]}
        
        return sample_dict

        
class Jeffreys:
    def __init__(self):

        self.ciao=1

        
    def aggregate_likelihood(self, data, number_of_lags:int=1, jumps:int=1):

        OLS_estimator = OLS_new_estimator(data, number_of_lags=number_of_lags, jumps=jumps)

        beta_1 = OLS_estimator["beta_OLS"]

        X_0 = data_to_X_0(data, number_of_lags, jumps)
        X_2 = data_to_X_2(data, number_of_lags, jumps)

        T = data.shape[1]
        p = number_of_lags * jumps

        S_1 = get_S_L_from_beta(beta_1, X_0, X_2)
        nu_1= T-p

        self.posterior_hyperparameters={"beta_1": beta_1,
                                        "S_1": S_1,
                                        "nu_1": nu_1,
                                        "X_0": X_0,
                                        "X_2": X_2}
        
    def sample(self, size)->dict[str, np.ndarray[np.float_]]:

        betas=[]
        Sigmas=[]

        nu_1 = self.posterior_hyperparameters["nu_1"]
        S_1 =self.posterior_hyperparameters["S_1"]
        beta_1 = self.posterior_hyperparameters["beta_1"].squeeze()
        X_0 = self.posterior_hyperparameters["X_0"]
        X_2 = self.posterior_hyperparameters["X_2"]

        S_1_inv = np.linalg.inv(S_1)

        sampled_Sigma_inv = sample_wishart(S_1_inv, nu_1, size)

        N = S_1.shape[0]

        X_0_kron_I_N = kron_identity(N, X_0)

        for Sigma_inv in sampled_Sigma_inv:
            
            Sigma = np.linalg.inv(Sigma_inv)

            Sigmas.append(Sigma)

            Xi_inv = identity_kron(nu_1, Sigma_inv)

            Lambda_inv = X_0_kron_I_N.T @ Xi_inv @ X_0_kron_I_N

            Lambda = np.linalg.inv(Lambda_inv)

            beta = np.random.multivariate_normal(beta_1, Lambda)

            betas.append(beta)

        sample_dict={"betas": betas,
                     "Sigmas": Sigmas}
        
        return sample_dict


        
def export_samples(loc: str, sample_dict: dict):
    keys = ["betas", "Sigmas"]

    for key in keys:
        block = np.concatenate(sample_dict[key], axis=0)

        block_df = pd.DataFrame(block)

        block_df.to_csv(loc+"\\"+key+".csv", index=False)


def read_samples(loc: str, dim: int, number_of_lags: int):
    parameters_dict = {"Parameters":    [], 
                       "Phi_0":         [], 
                       "Sigma":         []}

    
    block_df = pd.read_csv(loc+"\\Sigmas.csv")

    block = block_df.to_numpy()

    for i in range(block.shape[0]//dim):
        parameters_dict["Sigma"].append(block[i*dim:(i+1)*dim, :])

    
    block_df = pd.read_csv(loc+"\\betas.csv")

    block = block_df.to_numpy()

    beta_step =(dim*(1+dim*number_of_lags))

    for i in range(block.shape[0]//beta_step):
        beta = block[i*beta_step:(i+1)*beta_step, :].squeeze()

        phi_0, parameters=beta_to_parameters(beta=beta, dim=dim, number_of_lags=number_of_lags)

        parameters_dict["Phi_0"].append(phi_0)
        parameters_dict["Parameters"].append(parameters)


    return parameters_dict









