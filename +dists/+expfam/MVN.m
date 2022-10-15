classdef MVN < handle
    properties
        % hyperparameters
        dim
        % molments of prior
        mu_0
        Sigma_0  % Sigma_0 generally not used

        invSigma_0 
        invSigmamu_0  
        % model params
        mu
        Sigma
        invSigma
        invSigmamu
        
        E
        isUpdated
    end
    
    methods
        function self = MVN(mu_0,invSigma_0,mu,invSigma)
            self.mu_0 = mu_0;
            self.invSigma_0 = invSigma_0;
            self.Sigma_0 = inv(invSigma_0);
            self.invSigmamu_0 = invSigma_0*mu_0;  
            self.dim = length(mu_0);
            if ~exist('invSigma','var') 
                self.invSigma = invSigma_0; 
            else
                self.invSigma = invSigma; 
            end
            self.Sigma = inv(self.invSigma);
            if ~exist('mu','var')
                self.mu = sqrtm(self.Sigma)*randn(self.dim,1);
            else
                self.mu = mu;
            end
            self.invSigmamu = self.invSigma*self.mu;  
            
            self.setUpdated(false);
            self.isUpdated.logdetSigma_0 = false;
        end
        
        function rawupdate(self,data,p)  %%% CHECK THIS
            if(~exist('p','var'))
                p=ones(size(data,1),1);
            end
            idx=find(~isnan(sum(data,2)));
            n=sum(p(idx));
            Ex = p(idx)'*data(idx,:)/n;
            Exx = data(idx,:)'*bsxfun(@times,data(idx,:),p(idx))/n;
            self.updateSS(Ex',Exx',n);
        end
        
        function updateSS(self,Excoef,Exxcoef,n)  % operates on natural parameters
            if(n==0)
                self.invSigma = self.invSigma_0;
                self.Sigma = inv(self.invSigma);            
                self.mu = self.Sigma*self.invSigmamu_0;% + diag(sqrt(diag(self.Sigma)))*randn(size(self.mu));
                self.invSigmamu = self.invSigma_0*self.mu;
                self.setUpdated(false);
            else
                self.invSigma = self.invSigma_0 + n*Exxcoef;
                self.invSigmamu = self.invSigmamu_0 + n*Excoef;
                self.Sigma = inv(self.invSigma);            
                self.mu = self.Sigma*self.invSigmamu;
                self.setUpdated(false);                
            end
                        
        end
        
        function updateprior(self,mu_0,invSigma_0)
            self.isUpdated.logdetSigma_0 = false;
            self.isUpdated.KLqprior = false;
            self.mu_0=mu_0;
            self.invSigma_0=invSigma_0;
            self.invSigmamu_0 = invSigma_0*mu_0;
            self.Sigma_0 = inv(Sigma_0);
        end
        
        function setUpdated(self,bool)
            self.isUpdated.KLqprior = bool;
            self.isUpdated.entropy = bool;
            self.isUpdated.logdetSigma = bool;
            self.isUpdated.logZ = bool;
        end
                        
        function res = mean(self)
            res = self.mu;
        end
        
        function res = secondmoment(self)
            res = self.mu*self.mu' + self.Sigma;
        end
        
        function res = Ex(self)
            res = self.mu;
        end
        
        function res = EinvSigma(self)
            res = self.invSigma;
        end
        
        function res = ESigma(self)
            res = self.Sigma;
        end
        
        function res = Exx(self)
            res = self.mu*self.mu' + self.Sigma;
        end
        
        function res = entropy(self)
            res = self.dim/2*(log(2*pi)+1) + 1/2*self.logdetSigma;
        end
        
        function res = logZ(self)  % assumes natural parameter rep with 
                                   % sufficient statistics -x^2/2 and x
            if(self.isUpdated.logZ==false)
               self.E.logZ =  self.dim/2*log(2*pi)+1/2*self.logdetSigma ...
                   -1/2*self.mu'*self.invSigma*self.mu;
               self.isUpdated.logZ = true;
            end
            res = self.E.logZ;
        end
        
        function res = logdetSigma(self)  %avoid calling unless necessary, this shit ain't cheap.
            if(self.isUpdated.logdetSigma==false)
                L = chol(self.invSigma);
                self.E.logdetSigma = -2*sum(log(diag(L)));
                self.isUpdated.logdetSigma = true;
            end
            res = self.E.logdetSigma;
        end
        
        function res = ElogdetinvSigma(self)
            res = - self.logdetSigma;
        end
        
        function res = ElogdetSigma(self)
            res = self.logdetSigma;
        end
        
        function res = logdetSigma_0(self)  %avoid calling unless necessary, this shit ain't cheap.
            if(self.isUpdated.logdetSigma_0==false)
                L = chol(self.invSigma_0);
                self.E.logdetSigma_0 = -2*sum(log(diag(L)));
                self.isUpdated.logdetSigma_0 = true;
            end
            res = self.E.logdetSigma_0;
        end
        
        function res = logdetinvSigma(self)  %avoid calling unless necessary, this shit ain't cheap.
            if(self.isUpdated.logdetSigma==false)
                L = chol(self.invSigma);
                self.E.logdetSigma = -2*sum(log(diag(L)));
                self.isUpdated.logdetSigma = true;
            end
            res = -self.E.logdetSigma;
        end
        
        function res = logdetinvSigma_0(self)  %avoid calling unless necessary, this shit ain't cheap.
            if(self.isUpdated.logdetSigma_0==false)
                L = chol(self.invSigma_0);
                self.E.logdetSigma_0 = -2*sum(log(diag(L)));
                self.isUpdated.logdetSigma_0 = true;
            end
            res = -self.E.logdetSigma_0;
        end
        
        function res = KLqprior(self)
            if(self.isUpdated.KLqprior==false)
                self.E.KLqprior = 1/2*(trace(self.invSigma_0*self.Sigma) ...
                    + (self.mu_0-self.mu)'*self.invSigma_0*(self.mu_0-self.mu) ...
                    - self.dim + self.logdetSigma_0 - self.logdetSigma);
                self.isUpdated.KLqprior = true;
            end
            res = self.E.KLqprior;
        end
        
    end
end
