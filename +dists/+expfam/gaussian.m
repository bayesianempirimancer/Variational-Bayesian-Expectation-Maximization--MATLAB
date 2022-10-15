classdef gaussian < handle
    properties
        % hyperparameters
        mu_0
        nu_0 % precision prior
        % model params
        mu
        var
    end
    
    methods
        function self = gaussian(mu_0,nu_0,mu,var)
            self.mu_0 = mu_0;
            self.nu_0 = nu_0;
            
            if ~exist('mu','var'), self.mu = randn(size(mu_0)); else self.mu = mu; end
            if ~exist('var','var'), self.var = ones(size(mu_0)); else self.var = var; end
        end
        
        
        function update(self,y,rEgamma)
            self.var = 1/(self.nu_0 + sum(rEgamma));
            self.mu = self.var * (self.nu_0*self.mu_0 + sum(rEgamma.*y));
        end
                        
        function res = mean(self)
            res = self.mu;
        end
        
        function res = secondmoment(self)
            res = self.mu.^2 + self.var;
        end
        
        function res = expectlogjoint(self)
            res = 1/2*log(self.nu_0) - 1/2*log(2*pi) - ...
                1/2*self.nu_0*(self.secondmoment - 2*self.mean*self.mu_0 + self.mu_0^2);
        end
        
        function res = entropy(self)
            res = 1/2*log(2*pi*exp(1)*self.var);
        end
        
        function res = lowerboundcontrib(self)
            res = self.entropy() + self.expectlogjoint();
        end
        
        function res = KLqprior(self)
            res = (self.mu-self.mu0)^2/2/self.var ...
                + 1/2*(self.var*self.nu_0-1-log(1/self.nu_0/self.var));
        end
        
    end
end
