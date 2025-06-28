import numpy as np
from scipy.special import expit
from scipy.stats import norm, binom
import configparser
import argparse



class NormalSampler:
    def __init__(self, mu, sigma2):
        self.mu = mu
        self.sigma2 = sigma2
    def __mul__(self, other):
        if isinstance(other, NormalSampler):
            mu = np.array([self.mu, other.mu])
            sigma2 = np.array([self.sigma2, other.sigma2])
            return NormalSampler(mu, sigma2).colapse()
    
    def colapse(self):
        """
        如果mu, sigma2 是多維，這邊可以做乘法
        """
        sigma2 = 1 / (1 / self.sigma2).sum()
        loc = (self.mu / self.sigma2).sum() * sigma2
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
    
    def run_gibbs_sampling(self, B, alpha_prior: NormalSampler, beta_prior: NormalSampler):
        alpha_sampling = np.zeros(B+1)
        alpha_sampling[0] = alpha_prior.sample(1)[0]
        
        beta_sampling = np.zeros(B+1)
        beta_sampling[0] = beta_prior.sample(1)[0]
        

        #initialize cheating
        self.z_sampling = self.z
        
        for b in range(B):
            self.z_sampling = self._z_gibbs_sample(alpha_sampling[b], beta_sampling[b])        
            self.OLS_checker()
            alpha_sampling[b+1] = self._alpha_gibbs_sample(beta_sampling[b], self.z_sampling, alpha_prior)
            beta_sampling[b+1] = self._beta_gibbs_sample(alpha_sampling[b+1], self.z_sampling, beta_prior)
        
        self.alpha_sampling = alpha_sampling[1:]
        self.beta_sampling = beta_sampling[1:]
    
    def _alpha_gibbs_sample(self, beta, z, alpha_prior: NormalSampler):
        likelihood_mu = z[:, 1:] - z[:, :-1] - beta * np.exp(-(self.d[:,1:]-z[:,:-1])**2)
        
        likelihood = NormalSampler(likelihood_mu, np.ones_like(likelihood_mu) * self.sigma2).colapse()
    
        sampler = alpha_prior * likelihood
        
        return sampler.sample(1)[0]
    
    def _beta_gibbs_sample(self, alpha, z, beta_prior: NormalSampler):
        
        multiplier = np.exp(-(self.d[:,1:]-z[:,:-1])**2)
        
        likelihood_mu = (z[:, 1:] - z[:, :-1] - alpha) / multiplier
        sigma2 = self.sigma2 / multiplier**2
        
        likelihood = NormalSampler(likelihood_mu, sigma2).colapse()
        
        sampler = beta_prior * likelihood
    
        return sampler.sample(1)[0]
    
    def _z_gibbs_sample(self, alpha, beta):
        z = np.zeros((self.N, self.T))
        z[:,0] = np.random.normal(size=self.N)
        
        for t in range(1, self.T):
            z[:,t] = self._z_mh_sampling(alpha, beta, self.d[:, t] , z[:, t-1], self.y[:, t], self.z_sampling[:,t])
        
        return z
                
    def _z_mh_sampling(self, alpha, beta, d, z_t_1, y, z_prev):
        
        
        base_loc = z_t_1 + alpha + beta*np.exp(-(d-z_t_1)**2)
        base_sig = np.sqrt(self.sigma2)
        
        def norm_pdf(x):
            return norm.pdf(x, loc=base_loc, scale=base_sig)
        
        def pdf(x):
            return norm_pdf(x) * binom.pmf(y, 1, expit(x-d))
            
        z = norm.rvs(loc=base_loc, scale=base_sig)
        
        mh = pdf(z) * norm_pdf(z_prev) / pdf(z_prev) / norm_pdf(z)
        prob = np.minimum(mh, 1)
        
        result = np.where(np.random.rand(self.N) < prob, z, z_prev)
        
        return result
    
    def OLS_checker(self):
       X = (np.exp(-(self.d[:,1:] - self.z_sampling[:,:-1])**2)).reshape((-1,1))
       y = (self.z_sampling[:,1:] - self.z_sampling[:,:-1]).reshape(-1)
       
       ones = np.ones((X.shape[0], 1))
       X = np.hstack((ones, X))
    
       result, _, _, _ = np.linalg.lstsq(X, y)
       print(result)
       

       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default='DEFAULT')
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read('param.ini')
    param_config = config[args.param]
    
    alpha = param_config.getfloat('alpha')
    beta = param_config.getfloat('beta')
    sigma2 = param_config.getfloat('sigma2')
    
    model = Model3Exp(alpha, beta, sigma2)
    N = param_config.getint('N')
    T = param_config.getint('T')
    
    model.generate_data(N, T)
    
    
    B = param_config.getint('B')
    alpha_prior = NormalSampler(param_config.getfloat('alpha_prior_mu'), param_config.getfloat('alpha_prior_sigma2'))
    beta_prior = NormalSampler(param_config.getfloat('beta_prior_mu'), param_config.getfloat('beta_prior_sigma2'))
    model.run_gibbs_sampling(B, alpha_prior, beta_prior)
    
    print(f'true alpha: {alpha};  post mean: {model.alpha_sampling.mean()}, var: {model.alpha_sampling.var()}' )
    print(f'true beta: {beta};  post mean: {model.beta_sampling.mean()}, var: {model.beta_sampling.var()}' )
    breakpoint()
    