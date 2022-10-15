classdef gaussiandists < handle
    properties
        K
        gaussdists
        gamdists
    end
    
    methods
        function self = gaussiandists(K,mu_0,nu_0,alpha_0,beta_0,mu,var,alpha,beta)
            self.K = K;
            for k = 1:K
                if ~exist('mu','var')
                    self.gaussdists{k} = dists.expfam.gaussian(mu_0,nu_0);
                    self.gamdists{k} = dists.expfam.gammadist(alpha_0,beta_0);
                elseif ~exist('var','var')
                    self.gaussdists{k} = dists.expfam.gaussian(mu_0,nu_0,mu(k));
                    self.gamdists{k} = dists.expfam.gammadist(alpha_0,beta_0);
                else
                    self.gaussdists{k} = dists.expfam.gaussian(mu_0,nu_0,mu(k),var(k));
                    self.gamdists{k} = dists.expfam.gammadist(alpha_0,beta_0,alpha(k),beta(k));
                end
            end
            
        end
        
        function update(self,y,r)
            for k = 1:self.K
                r_k = r(k,:);
                
                self.gaussdists{k}.update(y, r_k*self.gamdists{k}.mean());
                
                self.gamdists{k}.update(sum(r_k), ...
                    sum(r_k .* (y.^2 - 2*y*self.gaussdists{k}.mean + ...
                    self.gaussdists{k}.secondmoment)));
            end
        end
        
        function res = expectloglike(self,y)
            res = zeros(self.K,numel(y));
            for k = 1:self.K
                gaussdist = self.gaussdists{k};
                gamdist = self.gamdists{k};
                res(k,:) = self.expectlogjoint(y,y.^2, ...
                    gaussdist.mean,gaussdist.secondmoment, ...
                    gamdist.mean,gamdist.loggeomean);
            end
        end
        
        function res = expectlogjoint(self,y,Ey2,Emu,Emu2,Egamma,Elngamma)
            res = 1/2*Elngamma - 1/2*log(2*pi) ...
                - 1/2*Egamma*(Ey2 - 2*Emu*y + Emu2);
        end
                
        function res = lowerboundcontrib(self)
            res = 0;
            for k = 1:self.K
                res = res + self.gamdists{k}.lowerboundcontrib();
                res = res + self.gaussdists{k}.lowerboundcontrib();
            end
        end
        
        function res = means(self)
            res = zeros(1,self.K);
            for k = 1:self.K
                res(k) = self.gaussdists{k}.mean();
            end
        end
        
        function res = variances(self)
            res = zeros(1,self.K);
            for k = 1:self.K
                res(k) = 1/self.gamdists{k}.mean();
            end
        end
    end
end
