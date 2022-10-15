classdef gamma < handle
    properties
        % hyperparameters
        alpha_0
        beta_0
        % model params
        alpha
        beta
    end
    
    methods
        function self = gamma(alpha_0,beta_0,alpha,beta)
            if nargin == 0, return; end
            
            self.alpha_0 = alpha_0;
            self.beta_0 = beta_0;
            
            if ~exist('alpha','var'), self.alpha = ones(size(alpha_0)); else self.alpha = alpha; end
            if ~exist('beta','var'), self.beta = ones(size(beta_0)); else self.beta = beta; end
        end
        
        function update(self,Ex,Elogx,n)  % all are expected to be vectors
            self.alpha = self.alpha_0 + n.*Elogx;
            self.beta = self.beta_0 + n.*Ex;
        end
                
        function updateSS(self,Ex,Elogx,n)  % all are expected to be vectors
            if(n>0)
                self.alpha = self.alpha_0 + n.*Elogx;
                self.beta = self.beta_0 + n.*Ex;
            else
                self.alpha = self.alpha_0;
                self.beta = self.beta_0;
            end
        end
        
        function res = mean(self)
            res = self.alpha ./ self.beta;
        end
        
        function res = var(self)
            res = self.alpha ./ self.beta.^2;
        end
        
        function res = secondmoment(self)
            res = (self.alpha.^2 + self.alpha)./self.beta.^2;
        end
        
        function res = loggeomean(self)
            res = psi(self.alpha) - log(self.beta);
        end
                
        function res = entropy(self)
            res = self.alpha - log(self.beta) + ...
                gammaln(self.alpha) + ...
                (1 - self.alpha).*psi(self.alpha);
            res = sum(res(:));
        end
        
        function res = logZ(self)
            res = -self.alpha.*log(self.beta) + gammaln(self.alpha);
        end
        
        function res = logZp(self)
            res = -self.alpha_0.*log(self.beta_0) + gammaln(self.alpha_0);
        end
        
        function res = expectlogjoint(self)
            res = self.alpha_0.*log(self.beta_0) - gammaln(self.alpha_0) + ...
                (self.alpha_0 - 1).*self.loggeomean() - ...
                self.beta_0.*self.mean();
            res = sum(res(:));
        end
        
        function res = lowerboundcontrib(self)
            res = self.entropy() + self.expectlogjoint();
        end

        function res = Eloglikelihood(self,data)  %returns a vector Ns X 1 
            % and expects a standard data matrix Ns x D.
            res = log(data)*(self.alpha-1) - data*self.beta;
                res = bsxfun(@plus,res,self.alpha.*log(self.beta)-gammaln(self.alpha));
                res(isnan(res)) = 0;
        end
        
        function res = KLqprior(self)  % returns a vector of KLqpriors.
            res = (self.alpha-self.alpha_0).*psi(self.alpha) - gammaln(self.alpha) + gammaln(self.alpha_0) ...
                + self.alpha_0.*(log(self.beta)-log(self.beta_0)) + self.alpha.*(self.beta_0./self.beta-1);
        end
    end
end
