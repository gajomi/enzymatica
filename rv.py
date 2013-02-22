

class RV(object):
    def __init__(self,space,Pfun):
        self.space = space
        self.Pfun = Pfun

class Normal(RV):
    def __init__(self,mu,sigma2):
        self.mu = mu
        self.sigma2 = sigma2
    
class Posterior(RV):
    def __init__(self,prior,model,data):
        self.prior = prior
        self.model = model
        self.data = data


