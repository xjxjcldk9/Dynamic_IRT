import numpy as np
from pydantic import BaseModel


class NormalPrior(BaseModel):
    mu: float
    sigma2: float



class NormalSampler:
    def __init__(self, mu, sigma2):
        self.mu = mu
        self.sigma2 = sigma2
    def __mul__(self, other):
        if isinstance(other, NormalSampler):
            post_mu = (self.mu * other.sigma2 + self.sigma2 * other.mu) / (self.sigma2 + other.sigma2)
            post_sigma2 = self.sigma2 * other.sigma2 / (self.sigma2 + other.sigma2)
            return NormalSampler(post_mu, post_sigma2)
    
    def sample(self, size):
        return np.random.normal(loc=self.mu, scale=np.sqrt(self.sigma2), size=size)
    
    def __repr__(self):
        return f"Normal distribution: mu={self.mu}, sigma2={self.sigma2}"
 
 

class Model3Exp:
    def __init__(self, alpha, beta, sigma2):
        self.alpha = alpha
        self.beta = beta 
        self.sigma2 = sigma2
        
    def generate_data(self, N, T):
        
        z = np.zeros((N,T))
        
        z[:,0] = np.random.normal(size=N)
        
        d = np.zeros((N,T))
        d[:,0] = np.random.normal(size=N)
        
        
        for t in range(1, T):
            d[:, t] = z[:, t-1] + np.random.normal(size=N)
            z[:, t] = self.alpha + self.beta * np.exp(-(d[:, t] - z[:, t-1])**2) + z[:, t-1] + np.random.normal(scale=np.sqrt(self.sigma2),
                                                                                                                size=N)
        p = 1 / (1 + np.exp(-(z - d)))
        y = np.random.binomial(n=1, p=p)
        
        self.z = z
        self.d = d
        self.y = y
        self.N = N
        self.T = T
    
    def run_gibbs_sampling(self, B, alpha_prior: NormalPrior, beta_prior: NormalPrior):
        pass 
    
    def _alpha_gibbs_sample(self, beta, d, z, alpha_prior: NormalPrior):
        alpha_prior = NormalSampler(alpha_prior.mu, alpha_prior.sigma2) 
        
        likelihood_mu = (beta * np.exp(-(d[:,1:]-z[:,:-1])**2) + z[:,:-1] - z[:,1:]).sum()
        likelihood = NormalPrior(likelihood_mu, self.sigma2 / (self.T-1) / self.N)
        sampler = alpha_prior * likelihood
        return sampler.sample(1)
    
    def _beta_gibbs(self):
        pass 
    
    def _z_gibbs(self):
        pass 




       
if __name__ == '__main__':
    model = Model3Exp(1,2,1)
    model.generate_data(100,20)
    breakpoint()