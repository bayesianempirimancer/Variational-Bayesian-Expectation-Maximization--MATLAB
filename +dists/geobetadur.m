classdef geobetadur < handle
    properties
        D % number of columns to return 
        N % number of rows
        
        gbdists % collection of geobeta distributions
    end
    
    methods
        function self = geobetadur(D,N,alpha_0,beta_0,alpha,beta)
            self.D = D;
            self.N = N;
            
            if ~exist('alpha','var')
                for n = 1:N
                    self.gbdists{n} = dists.geobeta(alpha_0,beta_0);
                end
            else
                for n = 1:N
                    self.gbdists{n} = dists.geobeta(alpha_0,beta_0, ...
                        alpha(n),beta(n));
                end
            end
        end
        
        function update(self,data)
            for n = 1:self.N
                self.gbdists{n}.update( sum(data(n,:)), ...
                    sum((1:self.D).*data(n,:)) );
            end
        end
        
        function res = Elambdavals(self)
            res = zeros(1,self.N);
            for n = 1:self.N
                res(n) = self.gbdists{n}.lambda.mean();
            end
        end
        
        function res = geomean(self)
            res = exp( self.loggeomean() );
        end
        
        function res = loggeomean(self)
            res = zeros(self.N,self.D);
            for n = 1:self.N
                res(n,:) = self.gbdists{n}.expectloglike(1:self.D);
            end
        end
                
        function res = lowerboundcontrib(self)
            res = 0;
            for n = 1:self.N
                res = res + self.gbdists{n}.lowerboundcontrib();
                lambda = self.gbdists{n}.lambda.mean();
                res = res - ((1 - lambda)*log2(1 - lambda) - lambda*log2(lambda))/lambda;
            end
            %res = res + 1/self.D*sum(sum( self.loggeomean() ));
        end
    end
end
