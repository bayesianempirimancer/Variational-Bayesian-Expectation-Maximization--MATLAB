classdef poissonnormal < handle
    
    properties
        dim % dim of poisson and mu
        eta % vector of independent normal distributions
        lambda
        % lower bound
        L
        
    end
    
    methods
        function self = poissonnormal(mu_0, invSigma_0, mu, invSigma)
            self.dim = length(mu_0);
            if ~exist('mu','var') || ~exist('invSigma', 'var')
                self.eta = dists.expfam.normal(mu_0,invSigma_0);
                self.lambda = exp(mu_0+1/2./invSigma_0);
            else
                self.eta = dists.expfam.normal(mu_0,invSigma_0,mu,invSigma);
                self.lambda=exp(mu+1/2./invSigma);
            end
            self.L = -Inf;
        end
        
        function set_prior(mu_0,invSigma_0,Sigma_0)
            self.eta.mu_0 = mu_0;
            self.eta.Sigma_0 = Sigma_0;
            self.eta.invSigma_0 = 1./Sigma_0;
        end
                
        function dualVB(self, y, N, eps)
            if(~exist('eps','var'))
               eps=1; 
            end
            invA =  1./(self.eta.invSigma_0 + N*self.lambda); 
            dF = N*self.eta.Sigma_0.*(self.lambda-y) - self.eta.mu_0 - 0.5*invA + log(self.lambda);
            d2F = N*self.eta.Sigma_0 + 1./self.lambda+ N*0.5*invA.^2;
            self.lambda = self.lambda - eps * dF./d2F;             
            tol = 0.000001;
            self.lambda(self.lambda<tol) = tol;            
        end

        function res = F(self,y,N)
            A   = self.eta.invSigma_0 + N*self.lambda;
            res = 1/2*self.eta.Sigma_0'*(self.lambda- y).^2 ...
                - self.eta.mu_0'*(self.lambda- y) ...
                - 0.5/N*sum(log(A)) ...
                + self.lambda'*(log(self.lambda) - 1);
        end
            
        function L = updateSS(self,y,N,niter,eps)
            if(~exist('niter','var'))
                niter = 5;
                eps=1;                
            end
            if(~exist('eps','var'))
                eps=1;                
            end
            for ii= 1: niter
                self.dualVB(y,N,eps);
            end
            self.eta.mu = self.eta.mu_0 - N*self.eta.Sigma_0.*(self.lambda - y);
            self.eta.invSigma = self.eta.invSigma_0 + N*self.lambda;
            self.eta.Sigma =  1./self.eta.invSigma;
            self.eta.invSigmamu = self.eta.invSigma.*self.eta.mu;
        end
        
        function L = rawupdate(self,y,niter,eps,p)  % assumes y is a standard TxN data matrix
            if(nargin < 3 || isempty(niter))
                niter = 5;
            end
            if(nargin < 4 || isempty(eps))
               eps=0.9999;
            end
            if(nargin < 5)
                N=size(y,1);
                self.updateSS(mean(y)',N,niter,eps);
                L = sum(sum(self.Eloglikelihood(y))) - self.KLqprior;
            else
                N=sum(p);
                self.updateSS((p*y)'/N,N,niter,eps);
                L = sum(p*self.Eloglikelihood(y)) - self.KLqprior;
            end
            self.L = L;
        end

        
        function res = mean(self,idx)
            if(~exist('idx','var'))
                res = exp(self.eta.mu+0.5*self.eta.Sigma);
            else
                res = exp(self.eta.mu(idx)+0.5*self.eta.Sigma(idx));
            end
        end
        
        function res = priormean(self)
            res = exp(self.eta.mu_0+0.5*self.eta.Sigma_0);
        end
        
        function res = variance(self)
            res = exp(2*self.eta.mu+self.eta.Sigma).*(exp(self.eta.Sigma)-1);
        end
        
        function res = priorvariance(self)
            res = exp(2*self.eta.mu_0+self.eta.Sigma_0).*(exp(self.eta.Sigma_0)-1);
        end
        
        function res = secondmoment(self)
            res = self.mean.^2.*(exp(self.eta.Sigma)-1);
        end
        
        function res = KLqprior(self)
            res = self.eta.KLqprior;
        end
        
        function res = KLqpriorvec(self)            
            res = self.eta.KLqpriorvec;
        end
        
        function res = Eloglikelihood(self,y)
            res = bsxfun(@times,y,self.eta.mu');
            res = bsxfun(@minus,res,exp(self.eta.mu+0.5*self.eta.Sigma)') ...
                - log(factorial(y));
        end
                        
        function [dExcoeff,dExxcoeff] = effloglikelihood(self)
            dExcoeff = self.eta.invSigmamu - self.eta.invSigmamu_0;
            dExxcoeff = self.eta.invSigma - self.eta.invSigma_0;            
        end
                  
    end
    
end

