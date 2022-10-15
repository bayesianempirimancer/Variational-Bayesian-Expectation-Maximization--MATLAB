classdef normalsparse < handle  
    properties
        
                        
        dim        
        mu
        invSigma
        Sigma
        
        a_0
        b_0
        a
        b        

    end  
    
    methods               
        function self = normalsparse(n,a_0,b_0) 
            
            self.dim = n;
            self.a_0 = a_0; self.b_0 = b_0; 

%            self.a = repmat(self.a_0,n,1); 
            self.a = repmat(self.a_0 + 1/2,self.dim,1); % these never change
            self.b = repmat(b_0,n,1);
            
            self.invSigma = diag(self.a./self.b);
            self.Sigma = inv(self.invSigma);
            self.mu = sqrtm(self.Sigma)*randn(n,1);
            
        end
                
        function res = mean(self)
            res = self.mu;
        end
        
        function res = secondmoment(self)
            res = self.mu*self.mu' + self.Sigma;
        end
        
        function res = var(self)
            res = self.Sigma;
        end        
        
        function res = Ealpha(self)
            res = self.a./self.b;
        end
        
        function res = invSigmamu(self)
            res=self.invSigma*self.mu;
        end
        
        function res = ESigma(self)
            res=inv(self.invSigma);
        end
                                
        function updateSS(self,EX,EXX,n) 
                                                
                self.invSigma = n*EXX + diag(self.Ealpha);
                self.Sigma = inv(self.invSigma);
                self.mu = self.Sigma*EX*n;

                self.b = self.b_0 + 1/2*diag(self.secondmoment);
                %self.a never changes
        end
                                                                               
        function res = KLqprior(self)            
            % this is negative of entropy of q and cross entropy <-q*log p>
            res = -self.dim/2 + 1/2*sum(self.Ealpha.*(self.mu.^2+diag(self.Sigma))) ...
                  + 1/2*self.logdetinvSigma - 1/2*sum(psi(self.a)-log(self.b));
            
            res  = res + sum((self.a-1).*psi(self.a) + log(self.b) - self.a - gammaln(self.a)) ...
                - self.dim*(self.a_0*log(self.b_0) - gammaln(self.a_0)) ...
                - (self.a_0 - 1)*sum(psi(self.a)-log(self.b)) ...
                + self.b_0*sum(self.a./self.b);
            
        end
             
        function res = logdetinvSigma(self)  %avoid calling unless necessary, this shit ain't cheap.
            L = chol(self.invSigma);
            res = 2*sum(log(diag(L)));
        end
        
    end
end
