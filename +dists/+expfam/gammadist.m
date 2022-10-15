classdef gammadist < handle
    properties
        % hyperparameters
        alpha_0
        beta_0
        % model params
        alpha
        beta
    end
    
    methods
        function self = gammadist(alpha_0,beta_0,alpha,beta)
            if nargin == 0, return; end
            
            self.alpha_0 = alpha_0;
            self.beta_0 = beta_0;
            
            if ~exist('alpha','var'), self.alpha = ones(size(alpha_0)); else self.alpha = alpha; end
            if ~exist('beta','var'), self.beta = ones(size(beta_0)); else self.beta = beta; end
        end
        
        
        function update(self,alphaupdate,betaupdate)
            self.alpha = self.alpha_0 + 1/2*alphaupdate;
            self.beta = self.beta_0 + 1/2*betaupdate;
        end
                
        function res = mean(self)
            res = self.alpha ./ self.beta;
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
        
        function res = expectlogjoint(self)
            res = self.alpha_0.*log(self.beta_0) - gammaln(self.alpha_0) + ...
                (self.alpha_0 - 1).*self.loggeomean() - ...
                self.beta_0.*self.mean();
            res = sum(res(:));
        end
        
        function res = lowerboundcontrib(self)
            res = self.entropy() + self.expectlogjoint();
        end

        function res = KLqprior(self)
            res = (self.alpha-self.alpha_0).*psi(self.alpha) - gammaln(self.alpha) + gammaln(self.alpha_0) ...
               + self.alpha_0.*(log(self.beta)-log(self.beta_0)) + self.alpha.*(self.beta_0./self.beta-1);
        end
        
        % Make a copy of a handle object.
        function new = copy(self)
            % Instantiate new object of the same class.
            new = feval(class(self));
            
            % Copy all non-hidden properties.
            p = properties(self);
            for i = 1:length(p)
                new.(p{i}) = self.(p{i});
            end
        end
    end
end
