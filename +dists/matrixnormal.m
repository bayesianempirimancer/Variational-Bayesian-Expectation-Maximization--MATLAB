classdef matrixnormal < handle  
    properties
        % note that this is part of the conjugate prior for the regression
        % coefficients when there are n regressors and p dimensional
        % observations, i.e. CP = matrixnormal(M,U,V)*Wishart(V)*Wishart(V)
        % where U is the covariance of the regressions and V is the 
        % covariance of the observation noise (residuals).  
        
        n
        p
        
        M  % is n x p matrix of normal random variables
        U  % is n x n symmetric positive definite
        V  % is p x p symmetric positive definite
        
        invU
        invV
        
        % log p(X) = -1/2 * trace( inv(V) * (X-M)^T * inv(U) * (X-M) ) 
        %            - n*p/2*log(2*pi) - n/2 * logdet(V) - p/2 * logdet(U)
        
        % E[ (X-M)^T * (X-M) ] = V*trace(U)
        % E[ (X-M) * (X-M)^T ] = U*trace(V)

        % E[ X * A * X^T ] = U * trace( A^T*V ) + M*A*M^T
        % E[ X^T * B * X ] = V * trace( U*B^T ) + M^T*B*M
        % E[ X * C * X ]   = U * C^T * V + M*C*M
        
        % Entropy = n*p/2*(1+log(2*pi)) + n/2 * logdet(V) + p/2 * logdet(U)
        
        % Eloglike(Y,X) will assume Y is a p by T matrix of T
        % observations and X is a n by T matrix of regressors.
        
    end  
    
    methods  % only the constructor deals with actual matrices all other routines expect
             % mu and invSigma and Sigma to be formatted in a manner consistent with the
             % temporary variables defined below.
             
        function self = matrixnormal(M_0,U_0,V_0)
            
            self.dims = size(M_0);
            for i=1:self.dims(1)
                mu_0{i}=M_0(i,:)';
                invSigmamu_0{i} = (invSigma_0in(i,:).*mu_0in(i,:))';
                mu{i}=mu_0in(i,:)'+randn(self.dims(2),1)./sqrt(invSigma_0in(i,:)');
                invSigmamu{i} = (invSigma_0in(i,:))'.*mu{i};
                for j=1:self.dims(1)
                    invSigma_0{i,j} = zeros(self.dims(2),self.dims(2));
                    invSigma{i,j} = zeros(self.dims(2),self.dims(2));
                    Sigma{i,j} = zeros(self.dims(2),self.dims(2));
                end                
                invSigma_0{i,i} = diag(invSigma_0in(i,:));
                invSigma{i,i} = invSigma_0{i,i};
                Sigma{i,i} = diag(1./invSigma_0in(i,:));
                self.idx{i}=self.dims(2)*(i-1)+1:self.dims(2)*i;
            end
                        
            muvec_0=zeros(self.dims(1)*self.dims(2),1);
            muvec=zeros(self.dims(1)*self.dims(2),1);
            invSigmavec_0=zeros(self.dims(1)*self.dims(2),self.dims(1)*self.dims(2));
            for i=1:self.dims(1)
%                muvec_0(self.it0(i):self.it1(i))=self.mu_0{i};
%                muvec(self.it0(i):self.it1(i))=self.mu{i};
                muvec_0(self.idx{i})=mu_0{i};
                muvec(self.idx{i})=mu{i};
                for j=1:self.dims(1)
                    invSigmavec_0(self.idx{i},self.idx{j}) = invSigma_0{i,j};
                end
            end
            
            self.MVNvec = dists.expfam.MVN(muvec_0,invSigmavec_0,muvec,invSigmavec_0);    
                        
        end
        
        function update(self,Excoef,Exxcoef)  % operates on natural parameters
            
            % Again the expectation here is that Excoef is Excoef{i}(k) and
            % Exxcoef is Exxcoef{i,j}(k,l)
            
            Exvec=zeros(size(self.dims(1)*self.dims(2)));
            Exxvec=zeros(size(self.dims(1)*self.dims(2)),size(self.dims(1)*self.dims(2)));
            for i=1:self.dims(1)
                Exvec(self.idx{i})= Excoef{i};
                for j=1:self.dim(1)
                    Exxvec(self.idx{i},self.idx{j})=Exxcoef{i,j};
                end
            end
            
            self.MVN.update(Exvec,Exxvec);
            
            for i=1:self.dims(1)
                self.mu{i} = self.MVN.mu(self.idx{i});
            end
            
            for i=1:self.dims(1)
                for j=1:self.dims(1)
                    self.invSigma{i,j} = self.MVN.invSigma(self.idx{i},self.idx{j});
                    self.Sigma{i,j} = self.MVN.Sigma(self.idx{i},self.idx{j});
                end
            end

        end
        
        function updateprior(self,mu_0,invSigma_0)  
            % this function assumes mu_0 and invSigma_0 take on the format
            % used in the properties section
            
            % push update to MVNvec
            muvec_0 = zeros(self.dims(1)*self.dims(2),1);
            invSigmavec_0 = zeros(self.dims(1)*self.dims(2),self.dims(1)*self.dims(2));
            
            for i=1:self.dims(1)
                muvec_0(self.idx{i})=mu_0{i};
                for j=1:self.dims(1)
                    invSigmavec_0(self.idx{i},self.idx{j})=invSigma_0{i,j}
                end
            end
            
            self.MVNvec.updateprior(muvec_0,invSigmavec_0)
            
        end
                                
        function res = mean(self,i)
%            res = self.mu{i};
            if ~exist('i','var')  % output as matrix.
                res = reshape(self.MVNvec.mu,self.dims(2),self.dims(1))';  % sorry for not thinking ahead.
            else
                res = self.MVNvec.mu(self.idx{i});
            end
        end
        
        function res = secondmoment(self,i,j)
%            res = self.mu{i}*self.mu{j}' + self.Sigma{i,j};
            res = self.MVNvec.mu(self.idx{i})*self.MVNvec.mu(self.idx{j})' + self.MVNvec.Sigma(self.idx{i},self.idx{j});
        end
        
        function res = Ex(self,i)
%            res = self.mu{i};
            res = self.MVNvec.mu(self.idx{i});
        end
        
        function res = Exx(self,i,j)
%            res = self.mu{i}*self.mu{j}' + self.Sigma{i,j};
            res = self.MVNvec.mu(self.idx{i})*self.MVNvec.mu(self.idx{j})' + self.MVNvec.Sigma(self.idx{i},self.idx{j});
        end
        
        function res = entropy(self)
            res = self.MVNvec.entropy;
        end
        
        function res = logdetSigma(self)  %avoid calling unless necessary, this shit ain't cheap.
            res = self.MVNvec.logdetSigma;
        end
        
        function res = logdetSigma_0(self)  %avoid calling unless necessary, this shit ain't cheap.
            res = self.MVNvec.logdetSigma_0;
        end
        
        function res = KLqprior(self)
            res = self.MVNvec.KLqprior;
        end
        
    end
end
