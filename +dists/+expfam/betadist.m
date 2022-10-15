classdef betadist < handle
    properties
        % hyperparameters
        alpha_0
        beta_0
        % model params
        alpha
        beta
    end
    
    methods
        function self = betadist(alpha_0,beta_0,alpha,beta)
            self.alpha_0 = alpha_0;
            self.beta_0 = beta_0;
            
            
            if ~exist('alpha','var'), self.alpha = self.alpha_0; else self.alpha = alpha; end
            if ~exist('beta','var'), self.beta = self.beta_0; else self.beta = beta; end
        end
        
        function update(self,alphaupdate,betaupdate)
            self.alpha = self.alpha_0 + alphaupdate;
            self.beta = self.beta_0 + betaupdate;
        end
        
        function updateSS(self,Ex,n)
            self.alpha = self.alpha_0 + Ex*n;
            self.beta = self.beta_0 + (1-Ex)*n;
        end
        
        function rawupdate(self,data,p)
            idx=find(~isnan(sum(data,2)));
            n=sum(p(idx));
            Ex = p(idx)'*data(idx,:);
            self.update(Ex',n-Ex');
        end
                
        function res = mean(self)
            res = self.alpha ./ (self.alpha + self.beta);
        end
        
        function res = loggeomean(self)
            res = psi(self.alpha) - psi(self.alpha + self.beta);
        end
        
        function res = loggeomeanmirror(self)
            res = psi(self.beta) - psi(self.alpha + self.beta);
        end
        
        function res = loglike(self)
            res = psi(self.alpha) - psi(self.beta);
        end
                
        function res = entropy(self)
            res = betaln(self.alpha,self.beta) - ...
                (self.alpha - 1).*psi(self.alpha) - ...
                (self.beta - 1).*psi(self.beta) + ...
                (self.alpha + self.beta - 2).*psi(self.alpha + self.beta);
        end
        
        function res = expectlogjoint(self)
            res = - betaln( self.alpha_0, self.beta_0 ) + ...
                (self.alpha_0 - 1).*self.loggeomean() + ...
                (self.beta_0 - 1).*self.loggeomeanmirror();
        end
        
        function res = lowerboundcontrib(self)
            res = self.entropy() + self.expectlogjoint();
        end
        
        function res = KLqprior(self)
            res = betaln(self.alpha_0,self.beta_0)-betaln(self.alpha,self.beta) ...
                + (self.alpha-self.alpha_0).*psi(self.alpha) + (self.beta-self.beta_0).*psi(self.beta) ...
                + (self.alpha_0-self.alpha + self.beta_0-self.beta).*psi(self.alpha+self.beta);
            res = sum(res);
        end
        
        function res = Eloglikelihood(self,data) % assume binary observations.
            res = data*(psi(self.alpha)-psi(self.alpha+self.beta)) ...
                + (1-data)*(psi(self.beta)-psi(self.alpha+self.beta));
            res(isnan(res)) = 0;
        end
    end
end
