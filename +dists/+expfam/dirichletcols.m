classdef dirichletcols < handle
    % here the dirichlet distributions of interest are stored as vectors
    % in a matrix of size dim x Ns where Ns is the number of data points.
    properties
        dim  % dim is the dimension of the number of states/labels
             % as used below Ns is the number of columns is free 
             % because it is the number of data points.  
        alpha_0
        alpha
    end
    
    methods
        function self = dirichletcols(dim,alpha_0,alpha)
            self.dim = dim;
            
            if ~exist('alpha_0','var')
                self.alpha_0 = ones(self.dim,1);
            else
                self.alpha_0 = alpha_0;
            end
            
            if ~exist('alpha','var')
                self.alpha = self.alpha_0.*(1+rand(size(self.alpha_0)));
            else
                self.alpha = alpha;
            end
        end
        
        function update(self,NA,beta)
            if ~exist('beta','var')
                beta = 1;
            end
            self.alpha = beta .* self.alpha_0 + NA;
        end
        
        function updateSS(self,NA)    %assumes NA is  dim x Ns
            self.alpha = bsxfun(@plus,self.alpha_0,NA);
        end
                        
        function res = mean(self)
            res = bsxfun(@rdivide,self.alpha,sum(self.alpha));
        end
                
        function res = loggeomean(self)
            res = bsxfun(@minus,psi(self.alpha),psi(sum(self.alpha)));
        end

        function res = variance(self)
           alpha_sum = sum(self.alpha);
           res = self.alpha.*bsxfun(@plus,alpha_sum,-self.alpha);
           res = bsxfun(@rdivide,res,alpha_sum.^2./(alpha_sum-1));
        end
                
        function res = entropy(self)
            alpha_sum = sum(self.alpha);
            res = sum(gammaln(self.alpha)) - gammaln(alpha_sum) + ...
                (alpha_sum - self.dim).*psi(alpha_sum) - ...
                sum((self.alpha - 1).*psi(self.alpha));
        end

        function res = KLqprior(self)
            alpha_sum = sum(self.alpha);
            res = gammaln(alpha_sum) - sum(gammaln(self.alpha)) ...
                - gammaln(sum(self.alpha_0)) + sum(gammaln(self.alpha_0)) ...
                + sum((self.alpha-self.alpha_0).*(psi(self.alpha)-psi(alpha_sum)));
            res = sum(res);
        end
        
        function res = logZ(self)
            res = sum(gammaln(self.alpha))-gammaln(sum(self.alpha));
            res = sum(res);
        end
        
        function res = logZp(self)
            res = size(self.alpha,2)*(self.dim*gammaln(self.alpha_0)-gammaln(self.dim*self.alpha_0));
        end
    end
end
