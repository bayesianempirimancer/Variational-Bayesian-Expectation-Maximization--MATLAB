classdef dirichletrows < handle
    properties
        K % number of categories 
        N % number of rows
        
        dirdists % collection of dirichlet distributions
    end
    
    methods
        function self = dirichletrows(K,N,dirdists,alpha_0)
            if nargin == 0, return; end
            
            self.K = K;
            self.N = N;
            
            if ~exist('dists','var')
                for n = 1:N
                    self.dirdists{n} = dists.expfam.dirichlet(K);
                end
            elseif iscell(dirdists)
                self.dirdists = dirdists;
            elseif ~exist('alpha_0','var')
                for n = 1:N
                    self.dirdists{n} = dists.expfam.dirichlet(K,ones(K,1),dirdists(n,:)');
                end
            else
                for n = 1:N
                    self.dirdists{n} = dists.expfam.dirichlet(K,alpha_0(n,:)'+eps,dirdists(n,:)');
                end
            end
        end
        
        
        
        function update(self,data,beta)
            if ~exist('data','var')
                data = 0;
            end
            if ~exist('beta','var')
                beta = 1;
            end
            
            for n = 1:self.N
                self.dirdists{n}.update(data(n,:)',beta);
            end
        end
        
        function res = mean(self)
            res = zeros(self.N,self.K);
            for n = 1:self.N
                res(n,:) = self.dirdists{n}.mean();
            end
        end
        
        function res = geomean(self)
            res = zeros(self.N,self.K);
            for n = 1:self.N
                res(n,:) = self.dirdists{n}.geomean();
            end
        end
        
        function res = loggeomean(self)
            res = zeros(self.N,self.K);
            for n = 1:self.N
                res(n,:) = self.dirdists{n}.loggeomean();
            end
        end
                
        function res = lowerboundcontrib(self,beta)
            if ~exist('beta','var')
                beta = 1;
            end
            res = 0;
            for n = 1:self.N
                res = res + self.dirdists{n}.lowerboundcontrib(beta);
            end
        end
    end
end
