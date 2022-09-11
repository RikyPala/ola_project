import numpy as np

class GaussianArm:
  
    def __init__(self, gamma_shape, gamma_rate, prior_mean):       
        self.n = 0 # the number of times this arm has been pulled
        self.x = [] # list of all samples
               
        self.alpha = gamma_shape # gamma shape parameter  
        self.beta = gamma_rate # gamma rate parameter  
      
        self.mu_0 = prior_mean # the prior (estimated) mean
        self.v_0 = self.beta / (self.alpha + 1) # the prior (estimated) variance
        
    def sample(self):  
        ''' sample from our estimated normal '''                   
        precision = np.random.gamma(self.alpha, 1/self.beta)
        if precision == 0 or self.n == 0: precision = 0.001
      
        estimated_variance = 1/precision
        return np.random.normal( self.mu_0, np.sqrt(estimated_variance))
                    
    def update(self, x):        
        ''' increase the number of times this arm has been pulles and improve the estimate of the 
            mean and variance by combining the single new value 'x' with the current estimate '''
        n = 1
        v = self.n
      
        self.alpha = self.alpha + n/2     
        self.beta = self.beta + ((n*v/(v + n)) * (((x - self.mu_0)**2)/2))
      
        # estimate the variance - calculate the mean from the gamma hyper-parameters
        self.v_0 = self.beta / (self.alpha + 1)      
            
        self.x.append(x)  # append the new value to the list of samples 
        self.n += 1
        self.mu_0 = np.array(self.x).mean()               



class TS:

    def __init__(self, n_arms, gamma_shape = 1, gamma_rate = 10, prior_mean = 1):
        self.arms = []
        for _ in range(n_arms): self.arms.append(GaussianArm(gamma_shape, gamma_rate, prior_mean))
        self.pulled = 0

    def pull(self):
        best_sample = 0
        for i in range(len(self.arms)):
            sample = self.arms[i].sample()
            if sample > best_sample:
                best_sample = sample
                self.pulled = i
        return self.pulled

    def update(self, reward):
        self.arms[self.pulled].update(reward)
