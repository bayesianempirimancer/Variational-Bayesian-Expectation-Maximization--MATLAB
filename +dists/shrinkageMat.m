classdef shrinkageMat < handle
    properties
        K % number of rows 
        J % number of columns
        Rk % number of region partitions among rows
        Rj % number of region partitions among columns
        Kr % number of rows per region (K = Rk*Kr)
        Jr % number of columns per region (J = Rj*Jr)
        
        mean
        variance
        marginalVariance
        varianceConditional
        
        alpha % local shrinkage
        beta % region shrinkage
    end
    
    methods
        function self = shrinkageMat(K,J,Rk,Rj,Kr,Jr,a,b,phi,mean)
            if nargin == 0, return; end
            
            self.K = K;
            self.J = J;
            self.Rk = Rk;
            self.Rj = Rj;
            self.Kr = Kr;
            self.Jr = Jr;
            
            if ~exist('mean','var')
                self.mean = eye(K,J);
            else
                self.mean = mean;
            end
            self.variance = ones(J,J,K);
            self.marginalVariance = ones(K,J);
            
            % place TPB prior on transformed local variance
            self.alpha = dists.TPB([K,J],phi,a,b,1);
            
            % place half-Cauchy prior on sqrt of region variance
            self.beta = dists.TPB([Rk,Rj],phi,1/2,1/2,1);
        end
        
        function update(self,mu,sig,sigConditional)
            % update A
            self.mean = mu;
            self.variance = sig;
            if exist('sigConditional','var'), self.varianceConditional = sigConditional; end
            for k = 1:self.K
                self.marginalVariance(k,:) = diag(sig(:,:,k));
            end
            
            % update alpha
            alphaScale = kron( self.beta.firstInvMoment, ones(self.Kr,self.Jr) );
            alphaUpdate = abs(alphaScale.*(self.mean.^2 + self.marginalVariance));
            self.alpha.update( alphaUpdate );
            
            self.alpha.mean = ones(self.K,self.J);
            self.alpha.firstInvMoment = ones(self.K,self.J);
            
            % update beta
            betaUpdate = zeros(self.Rk,self.Rj);
            for rk = 1:self.Rk
                for rj = 1:self.Rj
                    inds1 = ((rk-1)*self.Kr+1):(rk*self.Kr);
                    inds2 = ((rj-1)*self.Jr+1):(rj*self.Jr);
                    betaUpdate(rk,rj) = sum(sum(abs( ...
                        self.alpha.firstInvMoment(inds1,inds2) .* ...
                        (self.mean(inds1,inds2).^2 + self.marginalVariance(inds1,inds2)) )));
                end
            end
            self.beta.update( betaUpdate );
            
            %self.beta.mean = ones(self.Rk,self.Rj);
            %self.beta.firstInvMoment = ones(self.Rk,self.Rj);
        end
        
        function res = shrinkageMean(self)
            res = self.alpha.mean .* kron(self.beta.mean,ones(self.Kr,self.Jr));
        end
        
        function res = shrinkageInvMean(self)
            res = self.alpha.firstInvMoment .* ...
                kron(self.beta.firstInvMoment,ones(self.Kr,self.Jr));
        end
        
        function res = shrinkageLogGeoMean(self)
            res = self.alpha.logGeoMean + ...
                kron(self.beta.logGeoMean,ones(self.Kr,self.Jr));
        end
        
        function res = lowerboundcontrib(self)
            res = self.entropy() + self.expectlogjoint() + ...
                self.alpha.lowerboundcontrib() + ...
                self.beta.lowerboundcontrib();
        end
        
        function res = entropy(self)
            res = 0;
            for k = 1:self.K
                res = self.J/2*(1+log(2*pi)) + 1/2*log(det(self.variance(:,:,k)));
            end
        end
        
        function res = expectlogjoint(self)
            res = -self.K*self.J/2*log(2*pi) ...
                -1/2*sum(sum(self.shrinkageLogGeoMean)) ...
                -1/2* sum(sum(self.shrinkageInvMean() .* ...
                (self.mean.*self.mean + self.marginalVariance)));
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
