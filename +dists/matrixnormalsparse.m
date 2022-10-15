classdef matrixnormalsparse < handle  
    properties
        % Here we assume that n x p matrix X is sparse.  That is, 
        % X_{i,j}  normal(0,alpha_ij^{-1}) and the alpha_ij are gamma
        
        % the posterior is assumed to be of the form q(X|alpha)q(alpha)
        % Eloglikelihood assumes the problem is linear regression
                        
        dims
        n
        p
        
        mu
        invsig
        
        a_0
        b_0
        a
        b        

        invU
    end  
    
    methods               
        function self = matrixnormalsparse(n,p,a_0,b_0)  %neff>1
            
            self.dims = [n,p];
            self.n=n;
            self.p=p;
                        
            self.invsig = repmat(a_0/b_0,[n,p]);
            self.mu = a_0/b_0*rand(n,p);

            self.b_0 = b_0;
            self.a_0 = a_0;
            
            self.a = repmat(a_0+1/2,[n,p]);
            self.b = repmat(b_0,[n,p]);

            
            %used when there is a noise distribution for linear regression
            invU = dists.expfam.Wishart(eye(self.n)/self.n,self.n);
        end
        
        function res = Emu(self)
            res = self.mu;
        end
        
        function res = mean(self)
            res = self.mu;
        end
                                
%         function rawupdate(self,X,Y,p)  % X is p x T, Y is n x T and p is Tx1
%             idx=find(~isnan(sum(X,1)+sum(Y,1)));
%             n=sum(p(idx));
%             if(n>0)
%                 self.invV_0 = 
%                 self.invV = self.invV_0 + bsxfun(@times,X(:,idx),p(idx))*X(:,idx)';
%                 self.V = inv(self.invV);
%             
%                 self.mu = bsxfun(@times,Y(:,idx),p(idx))*X(:,idx)';
%                 self.mu = self.mu*self.V;
%             
%                 Exx = bsxfun(@times,Y(:,idx),p(idx))*Y(:,idx)' ...
%                     - self.mu*self.invV*self.mu' ...
%                     + self.mu_0*self.invV_0*self.mu_0';
%                 Exx = Exx/n;                
%                 self.invU.update(Exx,n);
%             else
%                 self.invV=self.invV_0;
%                 self.V=self.V_0;
%                 self.mu = self.mu_0;
%                 self.invU.update(0,0);
%             end
%         end
        
        function updateSS(self,EX,EXX,n) 
            if(n>0)
                
                self.invV = self.invV_0 + n*EXX;
                self.V = inv(self.invV);
            
                self.mu = self.mu_0*self.invV_0 + n*EYX;
                self.mu = self.mu*self.V;
            
                EYY = EYY*n - self.mu*self.invV*self.mu' ...
                    + self.mu_0*self.invV_0*self.mu_0';
                
                self.invU.update(EYY/n,n);            

                Bvec = reshape(n*EX,self.n*self.p,1);
                Phat = reshape(n*EXX,self.n*self.p,self.n*self.p);
                Phat = Phat + diag(reshape(self.a./self.b,self.n*self.p,1));

                self.mu = reshape(Phat\Bvec,self.n,self.p);
                self.invsig = reshape(diag(Phat),self.n,self.p);      
                self.a = self.a_0 + 1/2;
                self.b = self.b_0 + 1/2*(self.mu.^2 + 1./self.invsig);
            else
                self.mu = zeros(self.n,self.p);
                self.invsig = repmat(self.a_0/self.b_0,[self.n,self.p]);
                self.b = repmat(self.b_0,[self.n,self.p]);
            end

        end
                                                                       
        function res = entropy(self)   
            res = self.n*self.p/2*(1+log(2*pi)) - 1/2*sum(psi(self.a(:))-log(self.b(:))) ...
                  + sum(self.a(:)-log(self.b(:))+gammaln(self.a(:))+(1-a(:)).*psi(self.a(:)));
        end
        
        function res = KLqprior(self)            
            % this is negative of entropy of q and cross entropy <-q*log p>
            res = sum(self.a(:)./self.b(:).*self.mu(:).^2) ...
                - self.n*self.p*(self.a_0*log(self.b_0) - gammaln(self.a_0)) ...
                - (self.a_0 - 1)*sum(psi(self.a(:))-log(self.b(:))) ...
                + self.b_0*sum(self.a(:)./self.b(:));
        end
                        
        function res = Eloglikelihood(self,X,Y)
              res = nan;
        end
    end
end
