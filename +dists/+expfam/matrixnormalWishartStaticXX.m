classdef matrixnormalWishartStaticXX < handle  
    properties
        % Here we assume that  
        % (1) prior mean is zero(n,p)
        % (2) empirical mean of X is zero
        % (3) p << n
        % (4) prior on invU is parameterized by invUparm = 1/invU_0*eye(n)
        %     and nu_0 = n, so that the prior expected precision is eye(n)
        %
        % This allows us to use the matrix inversion lemma to greatly 
        % speed up the computation of the relevant inverses.  
        

        dims
        n
        p

        V_0  % is p x p and plays sort of plays the role of lambda_0 
             % in the normal wishart
        invV_0
        logdetinvV_0
        mu % is the conditional mean of the n x p matrix of normal random 
           % variables given invU 
        V
        invV
        logdetinvV
        
        SEYY
        invSEYY
        logdetSEYY
        traceinvSEYY
        
        nu_0
        nu
        invU_0

        muTmu
        EXTinvUX
        EXTinvU
        ElogdetinvU
        EtraceinvU
        EinvU
        
%        invU
    end  
    
    methods               
        function self = matrixnormalWishartStaticXX(n,p,U_0,V_0)  
            % note that U_0 here is a scalar
%             [n,p]=size(M_0);
%             self.mu_0 = M_0;
            self.dims(1) = n;
            self.dims(2) = p;
            self.n=n;
            self.p=p;            
                        
            self.V_0 = V_0;
            self.invV_0 = inv(V_0);
            self.invU_0 = 1/U_0;
            
            self.V = self.V_0;
            self.invV = self.invV_0;
            self.logdetinvV_0 = 2*sum(log(diag(chol(self.invV_0))));
            self.logdetinvV = self.logdetinvV_0;
            self.mu = sqrt(U_0)*randn(self.n,self.p)*sqrtm(V_0);
            self.muTmu = self.mu'*self.mu;
            
            %parms for Wishart distribution are just the number of data
            %points
            self.nu_0 = n+2;
            self.nu = n+2;
            alpha = self.invU_0;
%            temp = self.nu/alpha*(eye(self.p) + self.muTmu*inv(self.V*alpha - self.muTmu));
            temp = U_0*self.nu_0;
            self.EXTinvUX = temp*self.muTmu + self.V*self.n;
            self.EXTinvU  = temp*self.mu';                            
            self.ElogdetinvU = sum(psi((self.nu + 1 - [1:self.n])/2)) + self.n*log(2) ...
              - self.n*log(self.invU_0);    
            self.EtraceinvU = 1/self.invU_0*self.nu_0*self.n;
            self.SEYY=[];
            self.EinvU = eye(n)/self.invU_0*self.nu_0;
%            self.invU = dists.expfam.Wishart(self.invU_0*eye(self.n),self.nu_0);                                    
        end        
                        
        function res = logdet(self,A)
            res = 2*sum(log(diag(chol(A)))); 
        end
        
        function setmu(self,mu)
            self.mu = mu;
            self.muTmu = self.mu'*self.mu;
            self.nu=self.nu_0;
            self.EtraceinvU = self.n/self.invU_0*self.nu;         
            self.invV=self.invV_0;
            self.V=self.V_0;
            alpha = self.invU_0 + self.nu-self.nu_0;
            temp = self.nu/alpha*(eye(self.p) + self.muTmu*inv(self.V*alpha - self.muTmu));
            self.EXTinvUX = temp*self.muTmu + self.V*self.n;
            self.EXTinvU  = temp*self.mu';                            
            self.ElogdetinvU = sum(psi((self.nu + 1 - [1:self.n])/2)) + self.n*log(2) ...
              - self.n*log(alpha) - self.logdet(eye(self.p)-self.muTmu*self.invV/alpha); 
            self.EtraceinvU = self.nu/alpha*(self.n + trace(inv(alpha*self.V - self.muTmu)*self.muTmu));            
        end
        
        function setSEYY(self,SEYY)
            fprintf('Inverting Something Big....')
            self.SEYY = self.invU_0*eye(self.n) + SEYY;
            self.invSEYY = inv(self.SEYY);
            self.logdetSEYY = self.logdet(self.SEYY);
            self.traceinvSEYY = trace(self.invSEYY);    
            fprintf('done\n')
        end
        
        function updateSS(self,EXX,EYX,EYY,n) %EYY present for compatibility.  It is not used.
            if(isempty(self.SEYY))
                fprintf('Warning call setSEYY\n')
                return
            end
            if(n>0)
                self.nu = self.nu_0 + n;
                self.invV = self.invV_0 + n*EXX;
                self.V = inv(self.invV);  
                self.logdetinvV = 2*sum(log(diag(chol(self.invV))));
                self.mu = (n*EYX)*self.V;
                self.muTmu = self.mu'*self.mu;
            else
                self.nu = self.nu_0;
                self.invV=self.invV_0;
                self.V=self.V_0;
                self.logdetinvV=self.logdetinvV_0;
                self.mu = zeros(self.n,self.p);
                self.muTmu = self.mu'*self.mu;
            end
            
            
            invAmu = self.invSEYY*self.mu;
            muTinvAmu = self.mu'*invAmu;
            
            temp =inv(muTinvAmu-self.V);
            
            self.EXTinvU = self.nu*(invAmu'-muTinvAmu*temp*invAmu');
            
            self.EXTinvUX = self.n*self.V + self.EXTinvU*self.mu;
            
            self.ElogdetinvU = sum(psi((self.nu + 1 - [1:self.n])/2)) + self.n*log(2) ...
              - self.logdetSEYY - self.logdet(eye(self.p)-muTinvAmu*self.invV);
          
            self.EinvU = self.nu*(self.invSEYY - invAmu*(temp)*invAmu');
            self.EtraceinvU = trace(self.EinvU);
        end    
        
        function res = KLqprior(self)            
            % this is negative of entropy of q and cross entropy <-q*log p>
            %            alpha = self.invU_0 + self.nu - self.nu_0;                
            res =  self.n/2*self.logdetinvV - self.n/2*self.logdetinvV_0;
            res = res - self.n*self.p/2;
            res = res + 1/2*sum(sum(self.invV_0.*self.EXTinvUX));
            
            % above takes care of KLqprior on matrixnormaal part
            % below takes care of KLqprior on Wishart distribution part
            
%            logdetUparm = - self.n*log(alpha) - self.logdet(eye(self.p)-self.muTmu*self.invV/alpha);
%            ElogdetinvU = sum(psi((self.nu + 1 - [1:self.n])/2)) + self.n*log(2) + logdetUparm;
%            traceU = self.n/alpha + 1/alpha*trace(inv(alpha*self.V - self.muTmu)*self.muTmu);
%            self.ElogdetinvU = sum(psi((self.nu + 1 - [1:self.n])/2)) + self.n*log(2) ...
%              - self.n*log(alpha) - self.logdet(eye(self.p)-self.muTmu*self.invV/alpha); 
          
            logdetUparm = self.ElogdetinvU - sum(psi((self.nu + 1 - [1:self.n])/2)) - self.n*log(2);


            logZ = self.nu*self.n/2*log(2)  + self.n*(self.n-1)/4*log(pi) ...
                     + sum(gammaln((self.nu+1-[1:self.n])/2)) ...
                     + self.nu/2*(logdetUparm);
                 
            logZ_0 = self.nu_0*self.n/2*log(2) + self.n*(self.n-1)/4*log(pi) ...
                     + sum(gammaln((self.nu_0+1-[1:self.n])/2)) ...
                     - self.nu_0/2*self.n*log(self.invU_0);

             res = res - logZ + logZ_0 ...
                 + (self.nu-self.nu_0)/2*self.ElogdetinvU ...                 
                 -self.nu*self.n/2 + self.invU_0/2*self.EtraceinvU;
             
        end
        
    end
end
