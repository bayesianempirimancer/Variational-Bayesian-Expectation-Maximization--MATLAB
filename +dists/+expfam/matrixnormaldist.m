classdef matrixnormaldistSemp < handle  
    properties
        % note that this is part of the conjugate prior for linear regression
        % coefficients when there are n regressors and p dimensional
        % observations, i.e. CP = matrixnormal(M,U,V) where U and V
        % correspond to the row and column wise covariances.  Here Semp
        % stands for semi-empirial which means we assume invV_0 is
        % proportional to the empirical covariance matrix of the regressors
        % this allows us to avoid the need for iterative updates on U and V
        
        % This version of the routine assumes posterior of the form 
        % q(X|mu,invV,invU)
        
        % Note that V is just a parameter while invU is a wishart
        % distributed random variable.  
        dims
        n
        p

        mu_0    % is nxp matrix that gives the prior mean
        invV_0  % is normally a p x p, but her we replace it with a scalar that 
                % corresponds to effective number of data points associated
                % with the prior.  This is equivalent to assuming invV_0_matrix =
                % EXX*invV_0 
        invU_0  % prior on invU_0.  Could be an arbitrary matrix but in order to 
                % use the semi empirical trick we will have to scale it by
                % the number of data points.
                
        mu % is the conditional mean of the n x p matrix of normal random 
           % variables given invU 
        V
        U
        invV  
        invU  
                
        % log p(X) = -1/2 * trace( inv(V) * (X-M)^T * inv(U) * (X-M) ) 
        %            - n*p/2*log(2*pi) - n/2 * logdet(V) - p/2 * logdet(U)
        
        % E[ (X-M)^T * (X-M) | U,V ] = V*trace(U)
        % E[ (X-M) * (X-M)^T | U,V ] = U*trace(V)

        % E[ X * A * X^T | U,V ] = U * trace( A^T*V ) + M*A*M^T
        % E[ X^T * B * X | U,V ] = V * trace( U*B^T ) + M^T*B*M
        % E[ X * C * X | U,V ]   = U * C^T * V + M*C*M
        
        % E[ X^T * U^{-1} * X | U,V ] = n V  + M^T*U^{-1}*M
        % E[ X * V^{-1} * X^T | U,V ] = p U  + M^T*V^{-1}*M

        % Entropy | U,V = n*p/2*(1+log(2*pi)) + n/2 * logdet(V) + p/2 * logdet(U)
        
        % Eloglike(Y,X) will assume Y is a p by T matrix of T
        % observations and X is a n by T matrix of regressors.
        
        
    end  
    
    methods               
        function self = matrixnormaldistSemp(M_0,U_0,V_0)  %neff>1
            
            self.dims = size(M_0);
            self.n=self.dims(1);
            self.p=self.dims(2);
                        
            self.mu_0 = M_0;
            self.V_0 = V_0;
            self.invV_0 = inv(V_0);

            self.V = V_0;
            self.invV = self.invV_0;
            
            self.mu = M_0 + sqrtm(U_0)*randn(self.n,self.p)*sqrtm(V_0); 
            self.invU = dists.expfam.Wishart(U_0,self.n+2);                                    
        end
        
        function res = EXTAX(self,A)
            res = self.V * trace(self.invU.ESigma*A') + self.mu'*A*self.mu;            
        end
        
        function res = EXAXT(self,A)
            res = self.invU.ESigma * trace(A'*self.V) + self.mu*A*self.mu';            
        end
        
        function res = EXAX(self,A)
            res = self.invU.ESigma * A' * self.V + self.mu*A*self.mu;            
        end
        
        function res = EXXT(self)
            res = self.invU.ESigma * trace(self.V) + self.mu*self.mu';            
        end
        
        function res = EXTX(self)
            res = self.V * trace(self.invU.ESigma) + self.mu'*self.mu;            
        end

        function res = EXmMUTinvUXmMU(self)
            res = self.n * self.V;            
        end
        
        function res = EXmMUinvVXmMUT(self)
            res = self. p * self.invU.ESigma            
        end
        
        function res = EXTinvUX(self)
            res = self.n * self.V  + self.mu'*self.EinvU*self.mu;
        end
        
        function res = EXTinvU(self)
            res = self.mu'*self.EinvU;
        end
        
        function res = EXinvVXT(self)
            % E[ X * V^{-1} * X^T | U,V ] = p U  + M^T*<invV>*M
            res = self. p * self.invU.ESigma + self.mu*self.EinvV*self.mu';
        end
        
        function res = Emu(self)
            res = self.mu;
        end
        
        function res = mean(self)
            res = self.mu;
        end
        
        function res = EinvV(self)
            res = self.invV;
        end
        
        function res = EinvU(self)
            res = self.invU.mean;
        end
        
        function res = EU(self)
            res = self.invU.ESigma;
        end
        
        function res = EV(self)
            res = self.V;
        end
        
        function res = ElogdetinvU(self)  
            res = self.invU.Elogdet;
        end
                
        function res = logdetinvV(self)  
            L = chol(self.invV);
            res = 2*sum(log(diag(L)));
        end
                        
        function res = logdetinvV_0(self)  
            L = chol(self.invV_0);
            res = 2*sum(log(diag(L)));
        end
                        
        function rawupdate(self,X,Y,p)  % X is p x T, Y is n x T and p is 1xT
            Ns=size(X,2);
            if(~exist('p','var'))
                p=ones(1,Ns);
            end
            idx=find(~isnan(sum(X,1)+sum(Y,1)));
            n=sum(p(idx));
            if(n>0)
                self.invV = self.invV_0 + bsxfun(@times,X(:,idx),p(idx))*X(:,idx)';
                self.V = inv(self.invV);
            
                self.mu = self.mu_0*self.invV_0 + bsxfun(@times,Y(:,idx),p(idx))*X(:,idx)';
                self.mu = self.mu*self.V;
            
                Exx = bsxfun(@times,Y(:,idx),p(idx))*Y(:,idx)' ...
                    - self.mu*self.invV*self.mu' ...
                    + self.mu_0*self.invV_0*self.mu_0';
                Exx = Exx/n;                
                self.invU.update(Exx,n);            
            else
                self.invV=self.invV_0;
                self.V=self.V_0;
                self.mu = self.mu_0;
                self.invU.update(0,0);
            end
        end
        
        function updateSS(self,EXX,EYX,EYY,n,EXXupdate) 
            if(n>0)
                if(~exist('EXXupdate','var') || EXXupdate)
                    self.invV = self.invV_0 + n*EXX;
                    self.V = inv(self.invV);   % not needed if EXX never changes
                end
                self.mu = self.mu_0*self.invV_0 + n*EYX;
                self.mu = self.mu*self.V;
            
                EYY = EYY*n - self.mu*self.invV*self.mu' ...
                    + self.mu_0*self.invV_0*self.mu_0';
               
                self.invU.update(EYY/n,n);     
                
            else
                self.invV=self.invV_0;
                self.V=self.V_0;
                self.mu = self.mu_0;
                self.invU.update(0,0);                
            end

        end
                                                                       
        function res = entropy(self)   
                % Compute Expected entropy of the normal part, i.e.
                % N(mu,1/lambda*Sigma)

                res = self.n*self.p/2*(1+log(2*pi)) ...
                    - self.n/2*self.logdetinvV - self.p/2*self.invU.Elogdet;

                % Compute entropy of Wishart part
                
                res = res + self.invU.entropy;
        end
        
        function res = KLqprior(self)            
            % this is negative of entropy of q and cross entropy <-q*log p>
                        
            res = self.n/2*self.logdetinvV - self.n/2*self.logdetinvV_0;
            res = res - self.n*self.p/2;            
            res = res + 1/2*trace(self.invV_0*(self.mu-self.mu_0)'*self.invU.mean*(self.mu-self.mu_0));
            res = res + self.n/2*trace(self.invV_0*self.V);
            
            res = res + self.invU.KLqprior;
        end
                        
        function res = Eloglikelihood(self,X,Y)
            res = -self.n/2*log(2*pi) + 1/2*self.invU.meanlogdet;
            
            res = res -1/2*sum((self.invU.mean*(Y-self.mu*X)).*(Y-self.mu*X),1) ...
                  -1/2*self.n*sum((self.V*X).*X,1);
        end
        
        function res = copy(self)
            res = dists.expfam.matrixnormalWishart(self.mu_0,self.invU.V_0,self.V_0);
            res.mu = self.mu;
            res.V = self.V;
            res.invV = self.invV;
            res.invU.nu = self.invU.nu;
            res.invU.invV = self.invU.invV;
        end
    end
end
