import numpy as np
from .truncated_mvn_sampler import *
from scipy.stats import wishart, invwishart

class distribution:

    def __init__(self, hyperparameters: dict|None):
        self.hyperparameters=hyperparameters
        self.code="general"

    def sample(self, size=1):
        return None
    
class Deterministic_distribution(distribution):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.value=self.hyperparameters["Value"]
        self.code="Deterministic"

    def sample(self, size:int =1):
        
        return np.array([self.value for _ in range(size)])
    
class Normal_distribution(distribution):

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.mean=self.hyperparameters["Mean"]
        self.covariance_matrix=self.hyperparameters["Covariance_Matrix"]
        self.code="Normal"

    def sample(self, size=1):
        
        return np.random.multivariate_normal(self.mean, self.covariance_matrix, size=size)
    

class Uniform_distribution(distribution):

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.intervals=hyperparameters["Intervals"]
        self.low=hyperparameters["Intervals"][:,0].squeeze()
        self.high=hyperparameters["Intervals"][:,1].squeeze()
        self.dim=len(self.low)

        self.code="Uniform"

    def sample(self, size=1):

        return np.random.uniform(low=self.low, high=self.high, size=[size, self.dim])
    
class Truncated_Multivariate_Normal(distribution):

    def __init__(self, hyperparameters: dict | None):
        super().__init__(hyperparameters)

        self.mean=self.hyperparameters["Mean"]
        self.covariance_matrix=self.hyperparameters["Covariance_Matrix"]
        self.intervals=hyperparameters["Intervals"]
        self.low=hyperparameters["Intervals"][:,0].squeeze()
        self.high=hyperparameters["Intervals"][:,1].squeeze()
        self.dim=len(self.low)

        self.tmvn=TruncatedMVN(self.mean, self.covariance_matrix, self.low, self.high)

        self.code="Truncated_Multivariate_Normal"

    def sample(self, size=1):

        return self.tmvn.sample(size).T
    
def sample_inverse_wishart(inverse_scale, degree_of_freedom, size=1):

    # N = inverse_scale.shape[0]

    # X_Base = np.random.multivariate_normal(np.array([0 for _ in range(N)]), inverse_scale, size=(size, degree_of_freedom))

    # X_Wishart = [X.T @ X for X in X_Base]

    # X_inv_wishart = [np.linalg.inv(X) for X in X_Wishart]

    return invwishart.rvs(df=degree_of_freedom, scale= inverse_scale, size=size)#np.array(X_inv_wishart)

def sample_wishart(inverse_scale, degree_of_freedom, size=1):

    # N = inverse_scale.shape[0]

    # X_Base = np.random.multivariate_normal(np.array([0 for _ in range(N)]), inverse_scale, size=(size, degree_of_freedom))

    # X_Wishart = [X.T @ X for X in X_Base]

    return wishart.rvs(df=degree_of_freedom, scale= inverse_scale, size=size) #np.array(X_Wishart)





        

