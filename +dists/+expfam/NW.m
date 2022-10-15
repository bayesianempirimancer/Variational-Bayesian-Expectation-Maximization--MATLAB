classdef NW < handle
    properties
        dim % Dimension
        
        % prior parameters
        mu_0
        lambda_0
        V_0 
        invV_0
        logdetinvV_0
        nu_0

        % model params
        mu
        lambda
        V
        invV
        logdetinvV
        nu
        % E is where Expectations are stored, but they are only accessed
        % via functions.  
        E  % mu, mumu, Sigma, invSigma, logdetinvSigma, entropy, and KL(q,prior)
        isUpdated % are expectations up to date 
                  % (all set to false when a parameter update occurs)
    end
    
    methods
        function self = NW(mu_0,lambda_0,V_0,nu_0)
            self.mu_0 = mu_0;
            self.lambda_0 = lambda_0;
            self.nu_0 = nu_0;
            self.V_0 = V_0;
            self.invV_0 = inv(V_0); % a useful quantity to know.
            self.logdetinvV_0 = self.logdet(self.invV_0);
            
            self.dim = length(mu_0);
            self.mu = mu_0 + sqrtm(self.invV_0/(nu_0-self.dim-1))*randn(self.dim,1);
            
            self.lambda = lambda_0;
            self.nu = nu_0;
            self.invV = self.invV_0;
            self.V = V_0;       
            self.logdetinvV = self.logdetinvV_0;
            
            self.setUpdated(false);
        end
        
        function update(self,Ex,Exx,n) %Ex Exx are expected sufficient statistics
                                       %and n is the number of data points.
            if(n>0)
                self.lambda = self.lambda_0 + n;
                self.nu = self.nu_0 + n;
                self.mu = (self.lambda_0*self.mu_0 + n*Ex)/self.lambda;
                self.invV = self.invV_0 + self.lambda_0*self.mu_0*self.mu_0' ...
                    + n*Exx - self.lambda*self.mu*self.mu';
                self.V = inv(self.invV);
                self.logdetinvV = self.logdet(self.invV);
            else
                self.mu = self.mu_0;
                self.invV = self.invV_0;
                self.V = self.V_0;
                self.logdetinvV = self.logdetinvV_0;
                self.lambda = self.lambda_0;
                self.nu = self.nu_0;
            end
            self.setUpdated(false);

        end
                
        function rawupdate(self,data,p)
            idx=find(~isnan(sum(data,2)));
            n=sum(p(idx));
            Ex = p(idx)'*data(idx,:)/n;
            Exx = data(idx,:)'*bsxfun(@times,data(idx,:),p(idx))/n;
            self.updateSS(Ex',Exx',n);
        end
        
        function updateSS(self,Ex,Exx,n) %Ex Exx are expected sufficient statistics
                                       %and n is the number of data points.
            if(n>0)
                self.lambda = self.lambda_0 + n;
                self.nu = self.nu_0 + n;
                self.mu = (self.lambda_0*self.mu_0 + n*Ex)/self.lambda;
                self.invV = self.invV_0 + self.lambda_0*self.mu_0*self.mu_0' ...
                    + n*Exx - self.lambda*self.mu*self.mu';
                self.V = inv(self.invV);
                self.logdetinvV = self.logdet(self.invV);
            else
                self.mu = self.mu_0;
                self.invV = self.invV_0;
                self.V = self.V_0;
                self.logdetinvV = self.logdetinvV_0;
                self.lambda = self.lambda_0;
                self.nu = self.nu_0;
            end
            self.setUpdated(false);

        end
        
        function setUpdated(self,bool)            
            self.isUpdated.invSigma = bool;
            self.isUpdated.invSigmamu = bool;
            self.isUpdated.Sigma = bool;
            self.isUpdated.Sigmamu = bool;
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
            self.invV_0 = inv(V_0); % a useful quantity to know.      
            self.logdetinvV_0 = self.logdet(self.invV_0);
            self.dim = length(mu_0);
            self.setUpdated(false);
        end
                        
        function res = Emu(self)
            res = self.mu;
        end
        
        function res = mean(self)
            res = self.mu;
        end
        
        function res = var(self)
            res = diag(self.ESigma);
        end        
        
        function res = cov(self)
            res = self.Esigma;
        end
        
        function res = Emumu(self)
           	if(self.isUpdated.mumu==false)
                self.E.mumu=self.mu*self.mu' + self.ESigma/self.lambda;
                self.isUpdated.mumu=true;
            end
            res = self.E.mumu;
        end
        
        function res = ESigma(self)
           	if(self.isUpdated.Sigma==false)
                self.E.Sigma = self.invV/(self.nu - self.dim - 1);
                self.isUpdated.Sigma = true;
            end
            res  = self.E.Sigma;
        end
        
        function res = EinvSigma(self)
           	if(self.isUpdated.invSigma==false)
                self.E.invSigma = self.V*self.nu;
                self.isUpdated.invSigma = true;
            end
            res = self.E.invSigma;
        end

        function res = EinvSigmamu(self)
           	if(self.isUpdated.invSigmamu==false)
                self.E.invSigmamu = self.V*self.mu*self.nu;
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
                self.E.logdetinvSigma = sum(psi((self.nu + 1 - [1:self.dim])/2)) + self.dim*log(2) - self.logdetinvV;
                self.isUpdated.logdetinvSigma = true;
            end
            res = self.E.logdetinvSigma;
        end
                
        function res = entropy(self)   
            if(self.isUpdated.entropy==false)
                % Compute Expected entropy of the normal part, i.e.
                % N(mu,1/lambda*Sigma)
                res = self.dim/2*(1+log(2*pi/self.lambda)) - 1/2*self.ElogdetinvSigma;
                % Compute entropy of Wishart part
                logZ = self.nu*self.dim/2*log(2) - self.nu/2*self.logdetinvV + self.dim*(self.dim-1)/4*log(pi) ...
                     + sum(gammaln((self.nu+1-[1:self.dim])/2));
                 
                res = res + logZ ...
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
            
                % Do normal part first, this just involves means and
                % lambdas
                
                KLmu = self.dim/2*(self.lambda_0/self.lambda-1) + self.dim/2*log(self.lambda/self.lambda_0) ...
                    + 1/2*self.lambda_0*(self.mu-self.mu_0)'*self.V*self.nu*(self.mu-self.mu_0);

                % Do Wishart part next
                logZ = self.nu*self.dim/2*log(2) - self.nu/2*self.logdetinvV + self.dim*(self.dim-1)/4*log(pi) ...
                     + sum(gammaln((self.nu+1-[1:self.dim])/2));
                
                logZ_0 = self.nu_0*self.dim/2*log(2) - self.nu_0/2*self.logdetinvV_0 + self.dim*(self.dim-1)/4*log(pi) ...
                     + sum(gammaln((self.nu_0+1-[1:self.dim])/2));

                 self.E.KLqprior = (self.nu-self.nu_0)/2*self.ElogdetinvSigma ...
                    - 1/2*trace((self.invV-self.invV_0)*self.V*self.nu) - logZ + logZ_0 + KLmu;
                
                self.isUpdated.KLqprior = true;
            end
            res = self.E.KLqprior;
        end
                
        function res = Eloglikelihood(self,X) 
            % computes the expected log likelihood of each data point X
            % so summing computes total expected log likelihood 
            % it is assumed that X is a proper N x D data matrix
            % and that the likelihood is also gaussian.
            [N,D] = size(X);
            
            EinvSigma = self.V*self.nu;
            res = -1/2*sum((X*EinvSigma).*X,2) + X*EinvSigma*self.Emu ...
                  -1/2*sum(sum(self.Emumu.*EinvSigma)) ...
                  +1/2*self.ElogdetinvSigma - self.dim/2*log(2*pi);
            res(isnan(res)) = 0;
        end
        
        function res = ElogZ(self)  % computes the log of the partition function for the Wishart
            if(self.isUpdated.logZ==false)
                self.E.logZ = self.nu*self.dim/2*log(2) - self.nu/2*self.logdetinvV + self.dim*(self.dim-1)/4*log(pi) ...
                     + sum(gammaln((self.nu+1-[1:self.dim])/2));
                self.isUpdated.logZ = true;
            end
            res = self.E.logZ;
        end
        
        function res = EnormlogZ(self) % computes the expected log partition 
                                       % function of the normal part of the
                                       % normal wishart
            res = 1/2*sum(self.EinvSigmamu.*self.mu) - 1/2*self.ElogdetinvSigma ...
                +self.dim/2*log(2*pi);
        end
    end
    
end
