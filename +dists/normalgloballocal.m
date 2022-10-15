classdef normalgloballocal < handle  
    properties
        
                        
        dim        
        mu
        invSigma
        Sigma
        
        a_0
        b_0
        a
        b        

        c_0
        d_0
        c
        d

    end  
    
    methods               
        function self = normalgloballocal(n,a_0,b_0,c_0,d_0)  
            
            self.dim = n;
            self.a_0 = a_0; self.b_0 = b_0; self.c_0 = c_0; self.d_0 = d_0;   

            self.a = repmat(self.a_0,n,1); 
            self.c = self.c_0; 

            self.b = repmat(b_0,n,1);
            self.d = d_0;
            
            self.invSigma = diag(self.a./self.b*c_0/d_0);
            self.Sigma = inv(self.invSigma);
            self.mu = sqrt(self.invSigma)*randn(n,1);
            
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
        
        function res = Etau(self)
            res = self.c/self.d;
        end
                                
        function res = Ealpha(self)
            res = self.a./self.b;
        end
                                
        function updateSS(self,EX,EXX) % note that n may well be one in many cases
%            if(n>0)
                                                
                self.invSigma = EXX + self.Etau*diag(self.Ealpha);
                self.Sigma = inv(self.invSigma);
                self.mu = self.Sigma*EX;

                self.c = self.c_0 + self.dim/2; % these never change                
                self.d = self.d_0 + 1/2*sum(self.Ealpha.*diag(self.secondmoment));
                
                self.a = repmat(self.a_0 + 1/2,self.dim,1); % these never change
                self.b = self.b_0 + 1/2*diag(self.secondmoment)*self.Etau;

%                 
%             else
%                 self.mu = zeros(self.n,1);
%                 self.a = repmat(self.a_0,self.dim,1);
%                 self.b = repmat(self.b_0,self.dim,1);
%                 self.c = self.c_0;
%                 self.d = self.d_0;
%                 self.invSigma = eye(self.dim)*self.Etau*self.a_0/self.b_0;
%                 self.Sigma = eye(self.dim)/self.Etau/self.a_0*self.b_0;
%             end

        end
                                                                               
        function res = KLqprior(self)            
            % this is negative of entropy of q and cross entropy <-q*log p>
            res = -self.dim/2 + 1/2*self.Etau*sum(self.Ealpha.*(self.mu.^2+diag(self.Sigma))) ...
                  + 1/2*self.logdetinvSigma - self.dim/2*(psi(self.c)-log(self.d)) ...
                  - 1/2*sum(psi(self.a)-log(self.b));
            
            res  = res + sum((self.a-1).*psi(self.a) + log(self.b) - self.a - gammaln(self.a)) ...
                - self.dim*(self.a_0*log(self.b_0) - gammaln(self.a_0)) ...
                - (self.a_0 - 1)*sum(psi(self.a)-log(self.b)) ...
                + self.b_0*sum(self.a./self.b);
            
            res  = res + (self.c-1)*psi(self.c) + log(self.d) - self.c - gammaln(self.c) ...
                - (self.c_0*log(self.d_0) - gammaln(self.c_0)) ...
                - (self.c_0 - 1)*(psi(self.c)-log(self.d)) ...
                + self.d_0*sum(self.c./self.d);                        
        end
             
        function res = logdetinvSigma(self)  %avoid calling unless necessary, this shit ain't cheap.
            L = chol(self.invSigma);
            res = 2*sum(log(diag(L)));
        end
        
    end
end
