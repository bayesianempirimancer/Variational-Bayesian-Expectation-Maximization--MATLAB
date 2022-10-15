classdef poissgammadur < handle
    properties
        D % number of columns to return 
        N % number of rows
        
        pgdists % collection of poissgamma distributions
    end
    
    methods
        function self = poissgammadur(D,N,alpha_0,beta_0,alpha,beta)
            self.D = D;
            self.N = N;
            
            if ~exist('alpha','var')
                for n = 1:N
                    self.pgdists{n} = dists.poissgamma(alpha_0,beta_0);
                end
            else
                for n = 1:N
                    self.pgdists{n} = dists.poissgamma(...
                        alpha_0,beta_0,alpha(n),beta(n));
                end
            end
        end
        
        function update(self,data)
            for n = 1:self.N
                self.pgdists{n}.update( sum(data(n,:)), ...
                    sum((1:self.D).*data(n,:)) );
                %self.pgdists{n}.update( sum((1:self.D).*data(n,:)), ...
                %    sum(data(n,:)) );
            end
        end
        
        function res = Elambdavals(self)
            res = zeros(1,self.N);
            for n = 1:self.N
                res(n) = self.pgdists{n}.lambda.mean();
            end
        end
        
        function res = geomean(self)
            res = exp( self.loggeomean() );
        end
        
        function res = loggeomean(self)
            res = zeros(self.N,self.D);
            for n = 1:self.N
                res(n,:) = self.pgdists{n}.expectloglike(1:self.D);
            end
        end
                
        function res = lowerboundcontrib(self)
            res = 0;
            for n = 1:self.N
                res = res + self.pgdists{n}.lowerboundcontrib();
            end
        end
    end
end
