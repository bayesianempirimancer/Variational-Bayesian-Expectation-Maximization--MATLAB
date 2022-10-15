classdef poissgamma < handle
    properties
        lambda
    end
    
    methods
        function self = poissgamma(alpha_0,beta_0,alpha,beta)
            if ~exist('alpha','var')
                self.lambda = dists.expfam.gammadist(alpha_0,beta_0);
            else
                self.lambda = dists.expfam.gammadist( ...
                    alpha_0,beta_0,alpha,beta);
            end
        end
       
        function res = mean(self)
            res = self.lambda.mean;
        end
        
        function update(self,sumy,n)
            if(n>0)
                self.lambda.update( 2*sumy, 2*n );
            else
                self.lambda.alpha = self.lambda.alpha_0;
                self.lambda.beta = self.lambda.beta_0;
            end            
        end
        
        function rawupdate(self,data,p)
            if(~exist('p','var'))
                p=ones(size(data,1),1);
            end
            idx=find(~isnan(sum(data,2)));
            n=sum(p(idx));
            SEx = p(idx)'*data(idx,:);
            self.update(SEx',n);
        end

        function res = expectloglike(self,d)
            res = - sum(gammaln(d + 1),2) + d*self.lambda.loggeomean() - ...
                sum(self.lambda.mean());
        end
        
        function res = Eloglikelihood(self,data)  %returns a vector Ns X 1 
            % and expects a standard data matrix Ns x D.
            res = data*self.lambda.loggeomean - sum(self.lambda.mean) ...
                - sum(gammaln(data+1),2);
            res(isnan(res)) = 0;
        end
        
        function res = KLqprior(self)  % returns a vector of KLqpriors.
%            res = (self.lambda.alpha-self.lambda.alpha_0).*psi(self.lambda.alpha) - gammaln(self.lambda.alpha) + gammaln(self.lambda.alpha_0) ...
%                + self.lambda.alpha_0.*(log(self.lambda.beta)-log(self.lambda.beta_0)) + self.lambda.alpha.*(self.lambda.beta_0./self.lambda.beta-1);
            res = self.lambda.KLqprior;
        end

        function res = lowerboundcontrib(self)
            res = self.lambda.lowerboundcontrib();
        end
        
    end
end
