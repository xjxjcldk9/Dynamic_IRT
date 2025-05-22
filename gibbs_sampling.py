import numpy as np
from typing import NamedTuple
from scipy.special import expit
from scipy.stats import norm, binom


class NormalPrior(NamedTuple):
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
    
    def colapse(self):
        """
        如果mu, sigma2 是多維，這邊可以做乘法
        """
        sigma2 = 1 / (1 / self.sigma2).sum()
        loc = (self.mu / self.sigma2).sum() / sigma2
        return NormalSampler(loc, sigma2)
        

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
        p = expit(z-d)
        y = np.random.binomial(n=1, p=p)
        
        self.z = z
        self.d = d
        self.y = y
        self.N = N
        self.T = T
    
    def run_gibbs_sampling(self, B, alpha_prior: NormalPrior, beta_prior: NormalPrior):
        alpha_sampling = np.zeros(B+1)
        alpha_sampling[0] = NormalSampler(**alpha_prior).sample(1)
        
        beta_sampling = np.zeros(B+1)
        beta_sampling[0] = NormalSampler(**beta_prior).sample(1)
        
        for b in range(B):
            z_sampling = self._z_gibbs_sample(alpha_sampling[b])
            alpha_sampling[b+1] = self._alpha_gibbs_sample(beta_sampling[b], z_sampling, alpha_prior)
            beta_sampling[b+1] = self._beta_gibbs_sample(alpha_sampling[b+1], z_sampling, beta_prior)
        
        self.alpha_sampling = alpha_sampling[1:]
        self.beta_sampling = beta_sampling[1:]
    
    def _alpha_gibbs_sample(self, beta, z, alpha_prior: NormalPrior):
        likelihood_mu = (-beta * np.exp(-(self.d[:,1:]-z[:,:-1])**2) - z[:,:-1] + z[:,1:]).sum()
        likelihood = NormalPrior(likelihood_mu, self.sigma2 / (self.T-1) / self.N)
        sampler = NormalSampler(alpha_prior.mu, alpha_prior.sigma2)  * likelihood
        return sampler.sample(1)
    
    def _beta_gibbs_sample(self, alpha, z, beta_prior):
        
        multiplier = np.exp((self.d[:,1:]-z[:,:-1])**2)
        
        likelihood_mu = -(alpha+z[:,:-1]-z[:,1:]) * multiplier
        sigma2 = self.sigma2 * multiplier**2
        
        likelihood = NormalSampler(likelihood_mu, sigma2).colapse()
        sampler = NormalSampler(beta_prior.mu, beta_prior.sigma2)  * likelihood
        return sampler.sample(1)
    
    def _z_gibbs_sample(self, alpha, beta):
        z = np.zeros((self.N, self.T))
        z[:,0] = np.random.normal(size=self.N)
        
        for t in range(1, self.T):
            z[:,t] = self._z_mh_sampling(alpha, beta, self.d[:, t] , z[:, t-1], self.y[:, t])
        return z
            
            
    def _z_mh_sampling(self, alpha, beta, d, z_prev, y):
        base_loc = z_prev+alpha+beta+np.exp(-(d-z_prev)**2)
        base_sig = np.sqrt(self.sigma2)
        
        def norm_pdf(x):
            return norm.pdf(x, loc=base_loc, scale=base_sig)
        
        def pdf(x):
            return norm_pdf * binom.pmf(y, 1, expit(x-d))
            
        y = norm.rvs(loc=base_loc, scale=base_sig)
            
        mh = pdf(y) * norm_pdf(z_prev) / pdf(z_prev) / norm_pdf(y)
        probs = np.minimum(mh, 1)
        result = np.where(np.random.rand() < probs, z_prev, y)
        return result




       
if __name__ == '__main__':
    model = Model3Exp(1,2,1)
    model.generate_data(100,20)
    breakpoint()