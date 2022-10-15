classdef geobeta < handle
    properties
        lambda
    end
    
    methods
        function self = geobeta(alpha_0,beta_0,alpha,beta)
            if ~exist('alpha','var')
                self.lambda = dists.expfam.betadist(alpha_0,beta_0);
            else
                self.lambda = dists.expfam.betadist( ...
                    alpha_0,beta_0,alpha,beta);
            end
        end
        
        function update(self,n,sumx)
            self.lambda.update( n, sumx );
        end
        
        function res = expectloglike(self,d)
            res = (d - 1)*self.lambda.loggeomeanmirror() + ...
                self.lambda.loggeomean();
        end
        
        function res = lowerboundcontrib(self)
            res = self.lambda.lowerboundcontrib();
        end
    end
end
