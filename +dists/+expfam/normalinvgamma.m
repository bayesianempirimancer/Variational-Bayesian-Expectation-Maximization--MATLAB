classdef normalinvgamma < handle
    properties  %vector of univariate normalinversegammas
        % hyperparameters
        dim
        mu_0
        lambda_0 % precision prior
        % model params
        V_0
        invV_0
        nu_0
        logdetinvV_0
        
        mu
        lambda % precision prior
        % model params
        V
        invV
        nu
        logdetinvV
        isUpdated
        E
    end
    
    methods
        function self = normalinvgamma(mu_0,lambda_0,V_0,nu_0)
            self.dim=length(mu_0);
            self.mu_0 = mu_0;
            self.lambda_0 = lambda_0;
            self.nu_0 = nu_0;
            self.V_0 = V_0;
            self.invV_0 = 1./V_0; % a useful quantity to know.
            self.logdetinvV_0 = log(self.invV_0);
            
            self.mu = mu_0;% + sqrtm(self.invV_0/(nu_0 - self.dim -1))*randn(self.dim,1);
            self.lambda = lambda_0;
            self.nu = nu_0;
            self.invV = self.invV_0;
            self.V = V_0;       
            self.logdetinvV = self.logdetinvV_0;
            
            self.setUpdated(false);
        end
        
        function update(self,Ex,Exx,n)  % it is assumed that all these are vectors
       
            if(n==0)
                self.lambda = self.lambda_0;
                self.nu = self.nu_0;
                self.mu = self.mu_0;
                
                self.invV = self.invV_0;
                self.V = self.V_0;
                self.logdetinvV = self.logdetinvV_0;
            else
                self.lambda = self.lambda_0 + n;
                self.nu = self.nu_0 + n;
                self.mu = (self.lambda_0.*self.mu_0 + n.*Ex)./self.lambda;

                self.invV = self.invV_0 + self.lambda_0.*self.mu_0.^2 ...
                    + n.*Exx - self.lambda.*self.mu.^2;
                self.V = 1./self.invV;
                self.logdetinvV = log(self.invV);
            end
            self.setUpdated(false);

        end
        
        function rawupdate(self,data,p)
            idx=find(~isnan(sum(data,2)));
            n=sum(p(idx));
            Ex = p(idx)'*data(idx,:)/n;
            Exx = p(idx)'*(data(idx,:).^2)/n;    
            self.update(Ex',Exx',n)
        end
        
        function setUpdated(self,bool)            
            self.isUpdated.invSigma = bool;
            self.isUpdated.Sigma = bool;
            self.isUpdated.invSigmamu = bool;
            self.isUpdated.mumu = bool;
            self.isUpdated.logdetinvSigma = bool;
            self.isUpdated.entropy = bool;
            self.isUpdated.KLqprior = bool;
            self.isUpdated.logZ = bool;  
        end
                        
        function updateprior(self,mu_0,lambda_0,V_0,nu_0)
            self.mu_0 = mu_0;
            self.lambda_0 = lambda_0;
            self.nu_0 = nu_0;
            self.V_0 = V_0;
            self.invV_0 = 1./(V_0); % a useful quantity to know.      
            self.logdetinvV_0 = log(self.invV_0);

            self.setUpdated(false);
        end
                        
        function res = Emu(self)
            res = self.mu;
        end
        
        function res = mean(self)
            res = self.mu;
        end
        
        function res = var(self)
            res = self.ESigma;
        end
        
        function res = Emumu(self)
           	if(self.isUpdated.mumu==false)
                self.E.mumu = self.mu.^2 + self.ESigma./self.lambda;
                self.isUpdated.mumu = true;
            end
            res = self.E.mumu;
        end
        
        function res = ESigma(self)
           	if(self.isUpdated.Sigma==false)
                self.E.Sigma = self.invV./self.nu;
                self.isUpdated.Sigma = true;
            end
            res  = self.E.Sigma;
        end
        
        function res = EinvSigma(self)
           	if(self.isUpdated.invSigma==false)
                self.E.invSigma = self.V.*self.nu;
                self.isUpdated.invSigma = true;
            end
            res = self.E.invSigma;
        end

        function res = EinvSigmamu(self)
           	if(self.isUpdated.invSigmamu==false)
                self.E.invSigmamu = self.EinvSigma.*self.mu;
                self.isUpdated.invSigmamu = true;
            end
            res = self.E.invSigmamu;
        end
        
        function res = logdet(self,V)  %avoid calling unless necessary, this shit ain't cheap.
            L = chol(V);
            res = 2*sum(log(diag(L)));
        end

        function res = ElogdetinvSigma(self)  
           	if(self.isUpdated.logdetinvSigma==false)
                self.E.logdetinvSigma = psi(self.nu/2) + log(2) - self.logdetinvV;
                self.isUpdated.logdetinvSigma = true;
            end
            res = self.E.logdetinvSigma;
        end
                
        function res = entropy(self)   
            if(self.isUpdated.entropy==false)
                % Compute Expected entropy of the normal part, i.e.
                % N(mu,1/lambda*Sigma)
                res = 1/2*(1+log(2*pi./self.lambda)) - 1/2*self.ElogdetinvSigma;
                % Compute entropy of Wishart part
                logZ = self.nu/2*log(2) - self.nu/2.*self.logdetinvV +  ...
                     + gammaln(self.nu/2);
                 
                res = res + logZ ...
                    - self.nu/2*self.ElogdetinvSigma + self.nu/2;
                
                self.E.entropy = res;
                self.isUpdated.entropy = true;
            else
                res = self.E.entropy;
            end
        end
        
        function res = KLqprior(self)            
            % this is negative of entropy of q and cross entropy <-q*log p>
            if(self.isUpdated.KLqprior==false)
            
                % Do normal part first, this just involves means and
                % lambdas
                
                KLmu = 1/2*(self.lambda_0./self.lambda-1) + 1/2*log(self.lambda./self.lambda_0) ...
                    + 1/2*self.lambda_0.*(self.mu-self.mu_0).*self.EinvSigma.*(self.mu-self.mu_0);

                % Do Wishart part next
                logZ = self.nu/2*log(2) - self.nu/2.*self.logdetinvV  ...
                     + gammaln(self.nu/2);
                
                logZ_0 = self.nu_0/2*log(2) - self.nu_0/2.*self.logdetinvV_0 ...
                     + gammaln(self.nu_0/2);

                 self.E.KLqprior = (self.nu-self.nu_0)/2.*self.ElogdetinvSigma ...
                    - 1/2*(self.invV-self.invV_0).*self.EinvSigma - logZ + logZ_0 + KLmu;
                
                self.isUpdated.KLqprior = true;
            end
            res = self.E.KLqprior;
        end
                
        function res = Eloglikelihood(self,X) 
            % computes the expected log likelihood of each data point X
            % so summing computes total expected log likelihood 
            % it is assumed that X is a proper N x D data matrix
            [N,D] = size(X);
            
            res = -1/2*X.^2*self.EinvSigma + X*self.EinvSigmamu;
              
            res = res + sum(-1/2*self.Emumu.*self.EinvSigma+1/2*self.ElogdetinvSigma-1/2*log(2*pi));
            res(isnan(res)) = 0;
        end
        
        function res = ElogZ(self)  % computes the log of the partition function for the Wishart
            if(self.isUpdated.logZ==false)
                self.E.logZ = self.nu/2*log(2) - self.nu/2.*self.logdetinvV  ...
                     + gammaln(self.nu/2);
                self.isUpdated.logZ = true;
            end
            res = self.E.logZ;
        end
        
    end
end

