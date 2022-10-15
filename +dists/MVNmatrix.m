classdef MVNmatrix < handle  %COMMENT this shit would be so much easier with pointers
    properties
        % hyperparameters
        dims
        idx % idx{i} returns the indices of mvnmat that correspond to row i of the matrix

%         % molments of prior
%         mu_0
% 
% 
%         % natural parameters of prior which is assumed to be independent
%         % so invSigma_0 and invSigmamu_0 are both assumed to be matrices
%         % that are the same size as mu_0.
%         invSigma_0  
%         invSigmamu_0  
%         % model params
% 
%         mu  % mu{i} is assumed to be a vector so that 
%             % mu{i}(j) corresponds to the i,j th entry of the matrix
%         Sigma % Similarly Sigma{i,k}(j,l) is the covariance associated with  
%               % mu{i}(j) and mu{k}(l) 
%         invSigma % same structure as Sigma
%         invSigmamu % same structure as mu
        
        MVNvec % acutally does the hard work
               % everything else is just formatting.

    end
    
    methods  % only the constructor deals with actual matrices all other routines expect
             % mu and invSigma and Sigma to be formatted in a manner consistent with the
             % temporary variables defined below.
             
        function self = MVNmatrix(mu_0in,invSigma_0in)
            
            self.dims = size(mu_0in);
            for i=1:self.dims(1)
                mu_0{i}=mu_0in(i,:)';
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
