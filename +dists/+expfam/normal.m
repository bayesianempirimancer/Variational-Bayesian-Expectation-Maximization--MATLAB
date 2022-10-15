classdef normal < handle
    properties
        % hyperparameters
        dim
        mu_0
        Sigma_0
        invSigma_0 
        invSigmamu_0
        % model params
        mu
        invSigmamu
        invSigma
        Sigma
    end
    
    methods
        function self = normal(mu_0,invSigma_0,mu,invSigma)
            self.dim = length(mu_0);
            self.mu_0 = mu_0;
            self.invSigma_0 = invSigma_0;
            self.invSigmamu_0 = invSigma_0.*mu_0;
            self.Sigma_0 = 1./invSigma_0;
            
            if ~exist('mu','var')
                self.mu = randn(size(mu_0)).*sqrt(self.Sigma_0); 
            else
                self.mu = mu; 
            end
            if ~exist('invSigma','var')
                self.invSigma = invSigma_0;
            else
                self.invSigma = invSigma; 
            end
            
            self.invSigmamu = self.invSigma.*self.mu;
            self.Sigma = 1./self.invSigma;
        end
                        
        function res = mean(self)
            res = self.mu;
        end
        
        function res = secondmoment(self)
            res = self.mu.^2 + self.Sigma;
        end
        
        function updateSS(self,Excoef,Exxcoef,n)
            if(n==0)
                self.invSigma = self.invSigma_0;
                self.invSigmamu = self.invSigmamu_0;
                self.Sigma = 1./self.invSigma;            
                self.mu = self.Sigma.*self.invSigmamu;
            else
                self.invSigma = self.invSigma_0 + n*Exxcoef;
                self.invSigmamu = self.invSigmamu_0 + n*Excoef;
                self.Sigma = 1./self.invSigma;            
                self.mu = self.Sigma.*self.invSigmamu;
            end

        end
        
        function rawupdate(self,X,obsVar,p)
            n=sum(p);
            Exxcoef = 1./obsVar;
            Excoef = (p*X)'./obsVar/n;
            self.updateSS(Excoef,Exxcoef,n);
        end
                        
        function res = KLqprior(self)
            res = 1/2*self.invSigma_0'*(self.mu-self.mu_0).^2 ...
                + 1/2*self.Sigma'*self.invSigma_0 ...
                - self.dim/2 ...
                - 1/2*sum(log(self.Sigma.*self.invSigma_0));
        end
        
        function res = KLqpriorvec(self)
            res = 1/2*self.invSigma_0.*(self.mu-self.mu_0).^2 ...
                + 1/2*self.Sigma.*self.invSigma_0 ...
                - 1/2 ...
                - 1/2*(log(self.Sigma.*self.invSigma_0));
        end
        
    end
end
