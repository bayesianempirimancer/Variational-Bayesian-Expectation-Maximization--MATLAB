classdef transition < handle
    properties
        % distribution over a transition probability matrix A_ij which
        % gives the probabilty of a transition from i to j, thus sum(A,2)=1  
        dim
        alpha_0
        alpha
    end
    
    methods
        function self = transition(dim,alpha_0,A)
            self.dim = dim;
            
            if ~exist('alpha_0','var')
                self.alpha_0 = ones(self.dim,self.dim);
            else
                self.alpha_0 = alpha_0;
            end
            
            if ~exist('A','var')
                self.alpha = self.alpha_0.*(1+rand(size(self.alpha_0)));
            else
                self.alpha = A;
            end
        end

        function updateraw(self,data,p)
            if ~exist('data','var')
                data = 0;
            end
            if ~exist('p','var')
                p = ones(size(data,3),1);
            end
            idx=find(~isnan(sum(sum(data,2),1)));
            self.alpha = self.alpha_0+reshape(reshape(data(:,:,idx),self.dim^2,length(idx))*p(idx),self.dim,self.dim);
        end
        
        function update(self,data,beta)
            if ~exist('data','var')
                data = 0;
            end
            if ~exist('beta','var')
                beta = 1;
            end

            self.alpha = self.alpha_0*beta+sum(data,3);
        end
        
        function res = mean(self)
            res = bsxfun(@rdivide,self.alpha,sum(self.alpha,2));
        end
        
        function res = geomean(self)
            res = exp(self.loggeomean());
        end
        
        function res = loggeomean(self)
            res = bsxfun(@minus,psi(self.alpha),psi(sum(self.alpha,2)));
        end
        
        function res = KLqprior(self)
            alpha_sum = sum(self.alpha,2);  %dim x 1
            
            res = gammaln(alpha_sum) - sum(gammaln(self.alpha),2) ...
                - gammaln(sum(self.alpha_0,2)) + sum(gammaln(self.alpha_0),2) ...
                + sum((self.alpha-self.alpha_0).*bsxfun(@minus,psi(self.alpha),psi(alpha_sum)),2);
            res = sum(res);
            
        end
        
        %possibly broken due to change in alpha_0
%         
%         function res = entropy(self)
%             alpha_sum = sum(self.alpha,2);
%             res = sum(sum(gammaln(self.alpha),2) - gammaln(alpha_sum) + ...
%                 (alpha_sum - self.dim).*psi(alpha_sum) - ...
%                 sum((self.alpha - 1).*psi(self.alpha),2));
%         end
%         
%         function res = expectlogjoint(self,beta)
%             if ~exist('beta','var')
%                 alpha_prior = self.alpha_0;
%             else
%                 alpha_prior = self.alpha_0 .* beta;
%             end
%             res =  - self.dim * sum(gammaln(alpha_prior)) + ...
%                 self.dim * gammaln(sum(alpha_prior)) + ...
%                 sum((alpha_prior - 1).*sum(self.loggeomean(),1));
%         end
%         
%         function res = lowerboundcontrib(self,beta)
%             if ~exist('beta','var')
%                 beta = 1;
%             end
%             res = self.entropy() + self.expectlogjoint(beta);
%         end
    end
end
