classdef normalRowMat < handle
    properties
        K % number of rows 
        J % number of columns
        
        mean
        variance
        marginalVariance
        
        alpha % variance hyperparameter (size is K x J)
    end
    
    methods
        function self = normalRowMat(K,J,alpha,mean)
            if nargin == 0, return; end
            
            self.K = K;
            self.J = J;
            
            self.alpha = alpha;
            
            if ~exist('mean','var')
                self.mean = randn(self.K,self.J);
            else
                self.mean = mean;
            end
            self.variance = ones(self.J,self.J,self.K);
            self.marginalVariance = ones(self.K,self.J);
        end
        
        function update(self,mu,sig)
            self.mean = mu;
            self.variance = sig;
            for k = 1:self.K
                self.marginalVariance(k,:) = diag(sig(:,:,k));
            end
        end
                
        function res = lowerboundcontrib(self,varScale)
            if ~exist('varScale','var'), varScale = 1; end
            res = self.entropy(varScale) + self.expectlogjoint(varScale);
        end
        
        function res = entropy(self,varScale)
            res = 0;
            for k = 1:self.K
                res = self.K/2*(1+log(2*pi)) + ...
                    1/2*log(det(varScale(k)*self.variance(:,:,k)));
            end
        end
        
        function res = expectlogjoint(self,varScale)
            res = 0;
            for k = 1:self.K
                res = -self.J/2*log(2*pi) - 1/2*sum(log(self.alpha(k,:))) ...
                    -1/2* sum(self.alpha(k,:)'.^-1 .* ...
                    diag(self.mean(k,:)'*self.mean(k,:) + ...
                    varScale(k)*self.variance(:,:,k)));
            end
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
