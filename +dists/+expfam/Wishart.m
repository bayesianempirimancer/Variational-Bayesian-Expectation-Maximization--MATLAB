classdef Wishart < handle
    properties
        dim % Dimension
        
        % hyperparameters
        V_0 
        invV_0
        nu_0

        % model params
%        V
        invV
        nu
        % E is where Expectations are stored, but they are only accessed
        % via functions.  
        E  % Sigma, invSigma, logdetinvSigma, entropy, and KL(q,prior)
        isUpdated % are expectations up to date 
                  % (all set to false when a parameter update occurs)
    end
    
    methods
        function self = Wishart(V_0,nu_0)
            
            self.nu_0 = nu_0;
            self.V_0 = V_0;
            self.invV_0 = inv(V_0); % a useful quantity to know.
            
            self.dim = size(V_0,1);
            
            self.nu = nu_0;
            self.invV = self.invV_0;
%            self.V = V_0;       
            
            self.setUpdated(false);
            
        end
        
        function update(self,Exx,n)
            
            if(n>0)
                
                self.invV = self.invV_0 + n*Exx;
%                self.V = inv(self.invV);
                self.nu = self.nu_0 + n;
                self.setUpdated(false);
            else
                self.invV = self.invV_0;
%                self.V = self.invV_0;
                self.nu = self.nu_0;                
                self.setUpdated(false);
            end

        end
        
        function updateSS(self,Exx,n)
            
            if(n>0)
                
                self.invV = self.invV_0 + n*Exx;
%                self.V = inv(self.invV);
                self.nu = self.nu_0 + n;
                self.setUpdated(false);
            else
                self.invV = self.invV_0;
%                self.V = self.invV_0;
                self.nu = self.nu_0;                
                self.setUpdated(false);
            end

        end
        
        function setUpdated(self,bool)
            self.isUpdated.invSigma = bool;
            self.isUpdated.Sigma = bool;
            self.isUpdated.logdetinvSigma = bool;
            self.isUpdated.entropy = bool;
            self.isUpdated.KLqprior = bool;
            self.isUpdated.logZ = bool;            
        end
                        
        function updateprior(self,V_0,nu_0)
            self.nu_0 = nu_0;
            self.V_0 = V_0;
            self.invV_0 = inv(V_0); % a useful quantity to know.            
        end
                        
        function res = ESigma(self) % Sigma = inv(X) where X is the Wishart distributed random variable
           	if(self.isUpdated.Sigma==false)
                self.E.Sigma = self.invV/(self.nu - self.dim - 1);
                self.isUpdated.Sigma = true;
            end
            res  = self.E.Sigma;
        end
                
        function res = EinvSigma(self)
           	if(self.isUpdated.invSigma==false)
                self.E.invSigma = inv(self.invV)*self.nu;
                self.isUpdated.invSigma = true;
            end
            res = self.E.invSigma;
        end
        
        function res = logdetinvV(self)  %avoid calling unless necessary, this shit ain't cheap.
            L = chol(self.invV);
            res = 2*sum(log(diag(L)));
        end

        function res = logdetinvV_0(self)  %avoid calling unless necessary, this shit ain't cheap.
            L = chol(self.invV_0);
            res = 2*sum(log(diag(L)));
        end
        
        function res = mean(self)
            res = EinvSigma(self);
        end
        
        function res = EtraceSigma(self)
            res = trace(self.ESigma);
        end
        
        function res = EtraceinvSigma(self)
            res = trace(self.EinvSigma);
        end
        
        function res = meaninv(self)
            res = ESigma(self);
        end
        
        function res = meanlogdet(self)
            res = ElogdetinvSigma(self);
        end

        function res = ElogdetinvSigma(self)  
           	if(self.isUpdated.logdetinvSigma==false)
                self.E.logdetinvSigma = sum(psi((self.nu + 1 - [1:self.dim])/2)) + self.dim*log(2) - logdetinvV(self);
                self.isUpdated.logdetinvSigma = true;
            end
            res = self.E.logdetinvSigma;
        end
       
        function res = Elogdet(self)  
           	if(self.isUpdated.logdetinvSigma==false)
                self.E.logdetinvSigma = sum(psi((self.nu + 1 - [1:self.dim])/2)) + self.dim*log(2) - logdetinvV(self);
                self.isUpdated.logdetinvSigma = true;
            end
            res = self.E.logdetinvSigma;
        end
                
        function res = entropy(self)   
            if(self.isUpdated.entropy==false)

                logZ = self.nu*self.dim/2*log(2) - self.nu/2*self.logdetinvV + self.dim*(self.dim-1)/4*log(pi) ...
                     + sum(gammaln((self.nu+1-[1:self.dim])/2));
                 
                res = logZ ...
                    - (self.nu-self.dim-1)/2*self.ElogdetinvSigma + self.nu*self.dim/2;
                
                self.E.entropy = res;
                self.isUpdated.entropy = true;
            else
                res = self.E.entropy;
            end
        end
        
        function res = KLqprior(self)            
            % this is negative of entropy of q and cross entropy <-q*log p>
            if(self.isUpdated.KLqprior==false)
           
                logZ = self.nu*self.dim/2*log(2) - self.nu/2*self.logdetinvV + self.dim*(self.dim-1)/4*log(pi) ...
                     + sum(gammaln((self.nu+1-[1:self.dim])/2));
                
                logZ_0 = self.nu_0*self.dim/2*log(2) - self.nu_0/2*self.logdetinvV_0 + self.dim*(self.dim-1)/4*log(pi) ...
                     + sum(gammaln((self.nu_0+1-[1:self.dim])/2));

                 self.E.KLqprior = (self.nu-self.nu_0)/2*self.ElogdetinvSigma ...
                    - 1/2*trace((self.invV-self.invV_0)*self.EinvSigma) - logZ + logZ_0;
                
                self.isUpdated.KLqprior = true;
            end
            res = self.E.KLqprior;
        end
        
        function res = ElogZ(self)  % computes the log of the partition function for the Wishart
            if(self.isUpdated.logZ==false)
                self.E.logZ = - self.nu/2*self.logdetinvV + self.dim*self.nu/2+self.dim*(self.dim-1)*log(pi)/2 ...
                    + sum(gammaln((self.nu+1-[1:self.dim])/2));
                self.isUpdated.logZ = true;
            end
            res = self.E.logZ;

        end
        
    end
end

