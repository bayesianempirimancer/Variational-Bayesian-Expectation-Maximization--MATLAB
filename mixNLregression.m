classdef mixNLregression < handle

    % NOT FULLY tested
    % The idea here is to classify neuron by the shape of their non-linear
    % transfer functions.  The generative model is that each neuron has a
    % NLregression function describe in NLregression.m, but can have a
    % a unique filter function W that projects into that functino as well
    % as a unique baseline and sensitivity function
    
    % i.e. for neuron n, Y_n = a_n f_m(X*W_n) + b_n
    % where f_m is one of NC available non-linearities selected when latent z_n = m
    % Unlike a standard mixture of neural networks, this approach can
    % leverage the automatic occams razor effect of bayesian inference
    
    properties
        % Parameters
        NC  % number of clusters
        
        NN      % number of neurons
        Dmax    % maximum dimension of the nonlinearities considered
        DNL     % DNL(j) is the dimensionality of non-linearity j.
        NLC     % number clusters per non-linearity domain
        nn      % container for single neurons properties
                % if j is 1..NC
                % nn{n}.T     % is the number of data points 
                % nn{n}.NR    % is the number of regressors
                
                % nn{n}.W{j} = matrixnormal wishart 
                % nn{n}.gb{j} = multivariate normal
                % nn{n}.gamma{j}{k} = gamma.
                % nn{n}.psgz(j,k,t) gives the probability that the domain 
                %                   assignment was to element k for the t'th 
                %                   observation given non-linearity j.
                % nn{n}.lnpsgz(j,k,t) is the associated logptilde
                %
                % nn{n}.ugsz{j}{k}.mu(:,t) is the mean of u^t given 
                %                          domain assignment k               
                %
                % nn{n}.ugsz{j}{k}.Sigma(:,:) is the variance of u^t 
                %                 given domain assignment k, its the same 
                %                 for all t
                %
                % nn{n}.ugsz{j}{k}.invSigma(:,:,t) is the inverse 
                %                 variance of u^t given domain assignment k
                %
                % nn{n}.ugsz{j}{k}.invSigmamu(:,t) is the inverse 
                %                 variance of u^t given domain assignment k
                %                 times the mean.
                % 
                % nn{n}.NAsgz(j,k)   number of cluster assignments given z=j
                
       % Parameters that define the non-linearities         
        pi  % Dirichlet posterior over nonlinearity assignments prs
        rho % rho{j} is the dirichlet distribution over the domain 
            % selection variable s associated with each non-linearity j
        AB  % AB{j}{k} is DNL(j)+1 dim MVN for nonlinearity j and domain k 
        u   % u{j}{k} is DNL(j) dim normal for nonlinearity j and domain k 
            % and represents the domain location
        uSig % uSig{j} is the cluster size = eye(DNL(j))/NC^DNL(j)
        uinvSig % inverse of uSig{j} = eye(DNL(j))*NC^DNL(j)

        L % lower bound
                
        pz    % assignment probabilities (N x NC)
        lnpz  %
        NA    % number of assignments to each class        

    end
        
    methods
        function self = mixNLregression(NC,NLC,Dmax,alpha_0,NLalpha_0,data)
        % Initialize basic parameters
            self.NC=NC;  
            NN=length(data);
            self.NN = NN;
            self.NLC = NLC;
            self.Dmax = Dmax;            
            DNL = ceil(linspace(0,Dmax,NC+1));
            DNL = DNL(2:end)';
            self.DNL = DNL;
            self.L = -Inf;

        % Initialize posteriors on Non-Linearity
            self.pi=dists.expfam.dirichlet(self.NC,alpha_0/NC*ones(self.NC,1));
            for j=1:NC                
                self.rho{j} = dists.expfam.dirichlet(self.NLC,NLalpha_0/NLC*ones(self.NLC,1));
                self.uSig{j} = eye(DNL(j))/NLC^DNL(j);
                self.uinvSig{j} = eye(DNL(j))*NLC^DNL(j);                
                for k=1:NLC
                    self.AB{j}{k} = dists.expfam.MVN(randn(DNL(j)+1,1),eye(DNL(j)+1));
                    self.u{j}{k} =  dists.expfam.MVN(randn(DNL(j),1),eye(DNL(j)));
                end
            end
            
            % Initialize parameters for neurons
            for n=1:NN
                % assumes data{n}.Y and data{n}.X are in the standard T x D
                % format
                self.nn{n}.T = size(data{n}.X,1);
                self.nn{n}.NR = size(data{n}.X,2);                
                self.nn{n}.model{Dmax}=[];
                for j=1:NC
                    M0=zeros(self.nn{n}.NR,DNL(j));
                    U0=eye(self.nn{n}.NR);
                    V0=eye(DNL(j));
                    
                    self.nn{n}.W{j} = dists.expfam.matrixnormalWishart(M0,U0,V0);
                    self.nn{n}.gb{j} = dists.expfam.MVN([1;0;],eye(2));
                    for k=1:NLC
                        self.nn{n}.gamma{j}{k} = dists.expfam.gamma(1,1);
                    end
                end
            end
        end
        
        function L = update(self,data)
            if(self.L == -Inf)
                self.update_assignments(data);
            end
            self.update_NLparms();
            L = self.update_assignments(data);
        end
        
        function fillunusedZ(self,data,niters)
            idx=find(self.NA<1);
            if(isempty(idx))
                return
            end
            NE = length(idx);
            m = max(self.lnpz);            
            [m,idx2]=sort(m);
            subiters = ceil(niters/5);
            for i=1:NE
                n=idx2(i);
                j=idx(i);
                DNL = self.DNL(j);                
                self.fitnn(n,DNL,data{n}.Y,data{n}.X,niters);
                self.pullNLparms(n,j);
                self.update_suffstats(n,j,data{n}.Y,data{n}.X);
            end
            self.NA = sum(self.pz,2);    
            self.pi.update(self.NA);
        end
                
        function pullNLparms(self,n,j) 
            % pulls the NLparms out of fit from neuron n and puts them into
            % nonlinearity assuming the non-linearities are compatible in
            % dimension.  It also sets the gb to Normal([1;0],I/size(X,1))
            if(isempty(self.nn{n}.model{self.DNL(j)}))
                fprint('Warning: model dimension ',self.DNL(j), ' not fit for nn{',n,'}\n')
                fprint('Aborting Pull')
                return
            end
            
            self.nn{n}.gb{j}.mu(1,1)=1;
            self.nn{n}.gb{j}.mu(2,1)=0;
            self.nn{n}.gb{j}.invSigma = eye(2)*self.nn{n}.T;
            self.nn{n}.gb{j}.Sigma = eye(2)/self.nn{n}.T;
            self.nn{n}.gb{j}.invSigmamu = self.nn{n}.gb{j}.mu*self.nn{n}.T; 
                
            self.pz(:,n)=zeros(self.NC,1);
            self.pz(j,n) = 1;
            for k=1:self.NLC                                 
                self.AB{j}{k}.mu = self.nn{n}.model{self.DNL(j)}.AB{k}.mu';
                self.AB{j}{k}.invSigma = self.nn{n}.model{self.DNL(j)}.AB{k}.invV;
                self.AB{j}{k}.Sigma = inv(self.AB{j}{k}.invSigma);
                self.AB{j}{k}.invSigmamu = self.AB{j}{k}.invSigma*self.AB{j}{k}.mu;

                self.nn{n}.gamma{j}{k}.alpha = self.nn{n}.model{self.DNL(j)}.AB{k}.invU.nu/2;
                self.nn{n}.gamma{j}{k}.beta = self.nn{n}.model{self.DNL(j)}.AB{k}.invU.invV/2;

                self.u{j}{k}.invSigma = self.nn{n}.model{self.DNL(j)}.u{k}.invSigma;
                self.u{j}{k}.mu = self.nn{n}.model{self.DNL(j)}.u{k}.mu;
                self.u{j}{k}.invSigmamu = self.nn{n}.model{self.DNL(j)}.u{k}.invSigmamu;
                self.u{j}{k}.Sigma = self.nn{n}.model{self.DNL(j)}.u{k}.Sigma;
            end
            self.rho{j}.alpha = self.nn{n}.model{self.DNL(j)}.pi.alpha;
            self.nn{n}.W{j}.mu = self.nn{n}.model{self.DNL(j)}.W.mu;
            self.nn{n}.W{j}.V = self.nn{n}.model{self.DNL(j)}.W.V;
            self.nn{n}.W{j}.invV = self.nn{n}.model{self.DNL(j)}.W.invV;
            self.nn{n}.W{j}.invU.invV = self.nn{n}.model{self.DNL(j)}.W.invU.invV;
            self.nn{n}.W{j}.invU.nu = self.nn{n}.model{self.DNL(j)}.W.invU.nu;
            self.nn{n}.W{j}.invU.V = self.nn{n}.model{self.DNL(j)}.W.invU.V;                
            self.nn{n}.W{j}.invU.setUpdated(false);
            
        end
        
        function L = fitnn(self,n,dim,Y,X,iters);
            % uses the NLregression code to discover neuron n's specific
            % non-linearity and returns the lower bound
            if(isempty(self.nn{n}.model{dim}))
                self.nn{n}.model{dim} = NLregression(1,self.nn{n}.NR,dim,self.NLC,1);
            end
            i=0;
            while( i<iters )
                i=i+1;
                L = self.nn{n}.model{dim}.updateassignments(Y,X);
                self.nn{n}.model{dim}.updateparms(Y,X);
                if(i<0.2*iters)
                    self.nn{n}.model{dim}.fillunused(Y,X,0.25); 
                    self.nn{n}.model{dim}.updateassignments(Y,X);
                    self.nn{n}.model{dim}.updateparms(Y,X);
                end
            end
            self.nn{n}.model{dim}.clearlatents();
        end
        
        function fillunusedS(self,j,data,ndp)
            % Uses the best fit neuron to cluster j to try to fill unused
            % clusters in the domain of the non-linearity  
            % ndp is the number of poorly data points to look at when
            % making new clusters
            
            [m,n]=max(self.lnpz(j,:));
            [ELBO_contrib,psgz,lnpsgz,ugsz] = update_suffstats(self,n,j,data{n}.Y,data{n}.X);
            NA = sum(psgz,2);
            Sidx = find(NA<1);
        
            m = max(lnpsgz);
            [m,Tidx] = sort(m);            
            Tidx = Tidx(1:ndp);
            temp = [data{n}.Y(Tidx,1),data{n}.X(Tidx,:)*self.nn{n}.W{j}.mean];            
            z=kmeans(temp,length(Sidx));
            
            for i=1:length(Sidz)
                
            end
            
            
        end
        
        function fit(self,data,iters)
            self.update(data);
            for i=1:iters
                if(i<iters*0.25)
                    self.fillunusedZ(data,20);
                end
                self.update(data);
                for n=1:self.NN
                    [m,j]=max(self.pz(:,n));
                    self.plot(n,j,data{n}.Y,data{n}.X)
                end
                figure(self.NN+1), plot(i,self.L,'o'), hold on, drawnow               
            end
            figure(self.NN+1), hold off
            
        end
        
        function L = update_assignments(self,data)
                
            self.lnpz = repmat(self.pi.loggeomean,1,self.NN);
            for n=1:self.NN
            for j=1:self.NC
                % First update the latents conditioned on NLcluster label    
                self.lnpz(j,n) = self.lnpz(j,n) + ...
                    self.update_suffstats(n,j,data{n}.Y,data{n}.X);        
            end                
            end
            if(self.NC>1)
                self.pz = exp(bsxfun(@minus,self.lnpz,max(self.lnpz)));
                self.pz = bsxfun(@rdivide,self.pz,sum(self.pz,1));
                self.NA = sum(self.pz,2);
                idx = find(self.pz(:)>0);
                L = self.pz(:)'*self.lnpz(:) - self.pz(idx)'*log(self.pz(idx));
            else
                self.pz = ones(1,self.NN);
                self.NA = sum(self.pz);
                L = self.pz(:)'*self.lnpz(:);
            end

            for j=1:self.NC
            for k=1:self.NLC
                L = L - self.AB{j}{k}.KLqprior - self.u{j}{k}.KLqprior;
                for n=1:self.NN
                    L = L - self.nn{n}.gamma{j}{k}.KLqprior;
                end
            end
                L = L - self.rho{j}.KLqprior;
            end
            L = L - self.pi.KLqprior;
            self.L = L;
            
        end
        
        function [ELBO_contrib,psgz,lnpsgz,ugsz] = update_suffstats(self,n,j,Y,X)
            % Updates that latent variables associated with just one neuron
            % (n) and one non-linearity (j) using observed values Y and regressors X
            %
            % Quantities computed here are T, psgz, ugsz, NAgz
            %
            T = size(Y,1);            
            D = self.DNL(j);                            
            
            self.nn{n}.T = T;
            
            Wbar = self.nn{n}.W{j}.mean;
            WinvSigW = self.nn{n}.W{j}.EXTinvUX;
            XinvSig = self.nn{n}.W{j}.EinvU;
            logdetinvSigXX = self.nn{n}.W{j}.ElogdetinvU;
            gg = self.nn{n}.gb{j}.secondmoment;
            gb = gg(1,2);
            bb = gg(2,2);
            gg = gg(1,1);
            gbar = self.nn{n}.gb{j}.mean;
            bbar = gbar(2);
            gbar = gbar(1);
            
            lnpsgz = repmat(self.rho{j}.loggeomean,1,T);
            psgz = zeros(size(lnpsgz));
            
            for k=1:self.NLC
                Abar = self.AB{j}{k}.mean;
                Bbar = Abar(D+1);
                Abar = Abar(1:D);
                ABm2 = self.AB{j}{k}.secondmoment;                
                uu = self.u{j}{k}.secondmoment;
                
                gambar = self.nn{n}.gamma{j}{k}.mean;
                lngambar = self.nn{n}.gamma{j}{k}.loggeomean;
                
                invSigma = WinvSigW + self.uinvSig{j} + gambar*gg*ABm2(1:D,1:D);
                Sigma = inv(invSigma);
                
                temp = self.uinvSig{j}*self.u{j}{k}.mean - gambar*(gg*ABm2(1:D,D+1) + gb*Abar);
                
                invSigmamu = gambar*gbar*Abar*Y'+Wbar'*XinvSig*X';
                invSigmamu = bsxfun(@plus,invSigmamu,temp);
                mu = Sigma*invSigmamu;
                
                % Add in lnZpost 
                lnpsgz(k,:) = lnpsgz(k,:) + 1/2*sum(mu.*invSigmamu,1) ...
                            - 1/2*log(det(invSigma)) + D/2*log(2*pi);

                % Add everything that has no u in it  
                lnpsgz(k,:) = lnpsgz(k,:) - 1/2*sum(X'.*(XinvSig*X'),1) ...
                            - gambar/2*(Y.^2)' + gambar*(gbar*Bbar+bbar)*Y' ...
                            - gambar*gb*Bbar -1/2*gambar*gg*ABm2(D+1,D+1) ...
                            - gambar/2*bb ...
                            - 1/2*sum(uu(:).*self.uinvSig{j}(:)) ...
                            + 1/2*log(det(self.uinvSig{j})) - D/2*log(2*pi) ...
                            + 1/2*logdetinvSigXX - self.nn{n}.NR/2*log(2*pi) ...
                            + 1/2*lngambar - 1/2*log(2*pi);
                            
                        
                ugsz{k}.mu = mu;
                ugsz{k}.Sigma = Sigma;
                ugsz{k}.invSigma = invSigma;
                ugsz{k}.invSigmamu = invSigmamu;

            end

            psgz = exp(bsxfun(@minus,lnpsgz,max(lnpsgz)));
            psgz = bsxfun(@rdivide,psgz,sum(psgz,1));
                        
            NA = sum(psgz,2);
            self.nn{n}.NAsgz(j,:) = NA';
%            self.nn{n}.psgz(j,:,:) = psgz;
%            self.nn{n}.lnpsgz(j,:,:) = lnpsgz;
           
            self.nn{n}.SuXgz{j} = zeros(self.DNL(j),self.nn{j}.NR);
            self.nn{n}.Suugz{j} = zeros(self.DNL(j));
            self.nn{n}.Sugz{j} = zeros(self.DNL(j),1);
            
            for k=1:self.NLC
                temp = bsxfun(@times,ugsz{k}.mu,psgz(k,:));
                
                self.nn{n}.Suugsz{j}{k} = temp*ugsz{k}.mu' + NA(k)*ugsz{k}.Sigma;
                self.nn{n}.SuYgsz{j}{k} = temp*Y;
                self.nn{n}.Sugsz{j}{k}  = sum(temp,2);
                self.nn{n}.SYgsz{j}{k}  = psgz(k,:)*Y;
                self.nn{n}.SYYgsz{j}{k} = psgz(k,:)*Y.^2;
                
                self.nn{n}.SuXgz{j} = self.nn{n}.SuXgz{j} + temp*X;
                self.nn{n}.Suugz{j} = self.nn{n}.Suugz{j} + self.nn{n}.Suugsz{j}{k};
                self.nn{n}.Sugz{j}  = self.nn{n}.Sugz{j} + self.nn{n}.Sugsz{j}{k};
            end
                
            self.nn{n}.SXX = X'*X;
            idx=find(psgz(:)>0);
            self.update_Wgbgamma(n,j);
            ELBO_contrib = lnpsgz(:)'*psgz(:) - psgz(idx)'*log(psgz(idx)) ...
                    - self.nn{n}.W{j}.KLqprior - self.nn{n}.gb{j}.KLqprior;

            
        end
                
        function update_latents(self,data)
            for n=1:self.NN
            for j=1:self.NC
                self.update_Wgbgamma(n,j);
            end
            end
        end
        
        function update_Wgbgamma(self,n,j)
            % This updates the latent variables that are neurons specific, 
            % i.e. W, gamma, and gb
            % Do W and gb first, then gamma to pick up the residuals.
            % 
            % Compute sufficient statistics
            % 
            
            D = self.DNL(j);
            N=self.nn{n}.T;            
            EYY = self.nn{n}.SXX/N;
            EXX = self.nn{n}.Suugz{j}/N;
            EYX = self.nn{n}.SuXgz{j}/N;            
            self.nn{n}.W{j}.updateSS(EXX,EYX',EYY,N);
                            
            Exx = zeros(2);
            Ex = zeros(2,1);
            Ntot = 0;
            
            for k=1:self.NLC
                NA = self.nn{n}.NAsgz(j,k);
                gambar = self.nn{n}.gamma{j}{k}.mean;
                
                u1u1 = [self.nn{n}.Suugsz{j}{k},self.nn{n}.Sugsz{j}{k};self.nn{n}.Sugsz{j}{k}',NA;];
                Exx(1,1) = Exx(1,1) + gambar*sum(sum(u1u1.*self.AB{j}{k}.secondmoment));
                Exx(1,2) = Exx(2,1) + gambar*[self.nn{n}.Sugsz{j}{k},NA]*self.AB{j}{k}.mean;
                Exx(2,1) =Exx(1,2);
                Exx(2,2) = Exx(2,2) + gambar*NA;
                
                Ex(1,1) = Ex(1,1) + gambar*self.AB{j}{k}.mean'*[self.nn{n}.SuYgsz{j}{k};self.nn{n}.SYgsz{j}{k};];
                Ex(2,1) = Ex(2,1) + gambar*self.nn{n}.SYgsz{j}{k};
                Ntot = Ntot + NA;
            end
            
            self.nn{n}.gb{j}.updateSS(Ex,Exx,Ntot);
            
            gbm2 = self.nn{n}.gb{j}.secondmoment;
            gb = self.nn{n}.gb{j}.mean;
                
            for k=1:self.NLC
                NA = self.nn{n}.NAsgz(j,k);
                u1u1 = [self.nn{n}.Suugsz{j}{k},self.nn{n}.Sugsz{j}{k};self.nn{n}.Sugsz{j}{k}',NA;];
                Sgam = self.nn{n}.SYYgsz{j}{k} ...
                     + gbm2(1,1)*sum(sum(u1u1.*self.AB{j}{k}.secondmoment)) ...
                     + gbm2(2,2)*NA ...
                     + 2*gbm2(1,2)*self.AB{j}{k}.mean'*[self.nn{n}.Sugsz{j}{k};NA;] ...
                     - 2*gb(1)*self.AB{j}{k}.mean'*[self.nn{n}.SuYgsz{j}{k};self.nn{n}.SYgsz{j}{k};] ...
                     - 2*gb(2)*self.nn{n}.SYgsz{j}{k};
                 
                if(Sgam<0) 
                    stop
                end
                self.nn{n}.gamma{j}{k}.updateSS(Sgam/2/NA,1/2,NA);
            end
        end
        
        function update_NLparms(self)
            % Compute Expectations
            
            N=zeros(self.NC,self.NLC);
            for j=1:self.NC
            for k=1:self.NLC
                N(j,k) = 0;
                Sxx = zeros(self.DNL(j)+1);
                Sx =  zeros(self.DNL(j)+1,1);
                Suu = zeros(self.DNL(j));
                Su =  zeros(self.DNL(j),1);
                for n=1:self.NN                 
                    NA = self.nn{n}.NAsgz(j,k);
                    p = self.pz(j,n);
                    N(j,k) = N(j,k) + NA*p;
                    gambar = self.nn{n}.gamma{j}{k}.mean;
                    gbm2 = self.nn{n}.gb{j}.secondmoment;
                    gb = self.nn{n}.gb{j}.mean;                    
                    
                    Suu = Suu + p*self.uinvSig{j}*NA;
                    Su = Su + p*self.uinvSig{j}*self.nn{n}.Sugsz{j}{k};
                    
                    Sxx = Sxx + gambar*gbm2(1,1)*p*[self.nn{n}.Suugsz{j}{k},self.nn{n}.Sugsz{j}{k};self.nn{n}.Sugsz{j}{k}',NA;];
                    Sx = Sx - p*gambar*gbm2(1,2)*[self.nn{n}.Sugsz{j}{k};NA;] ...
                       + p*gambar*gb(1)*[self.nn{n}.SuYgsz{j}{k};self.nn{n}.SYgsz{j}{k};];

                end
                self.AB{j}{k}.updateSS(Sx/N(j,k),Sxx/N(j,k),N(j,k));
                self.u{j}{k}.updateSS(Su/N(j,k),Suu/N(j,k),N(j,k));
            end
                self.rho{j}.update(N(j,:)');
            end
            self.pi.update(sum(self.pz,2));
            
        end
            
        function [Y,U,psgz] = getPredictions(self,n,j,X)

            T=size(X,1);
            D=self.DNL(j);
                        
            Wbar = self.nn{n}.W{j}.mean;
            WinvSigW = self.nn{n}.W{j}.EXTinvUX;
            XinvSig = self.nn{n}.W{j}.EinvU;
            logdetinvSigXX = self.nn{n}.W{j}.ElogdetinvU;
            
            lnpsgz = repmat(self.rho{j}.loggeomean,1,T);
            psgz = zeros(size(lnpsgz));
            
            for k=1:self.NLC
                uu = self.u{j}{k}.secondmoment;                
                invSigma = WinvSigW + self.uinvSig{j};
                Sigma = inv(invSigma);
                
                temp = self.uinvSig{j}*self.u{j}{k}.mean;                
                invSigmamu = Wbar'*XinvSig*X';
                invSigmamu = bsxfun(@plus,invSigmamu,temp);
                mu = Sigma*invSigmamu;
                
                % Add in lnZpost 
                lnpsgz(k,:) = lnpsgz(k,:) + 1/2*sum(mu.*invSigmamu,1) ...
                            - 1/2*log(det(invSigma)) + D/2*log(2*pi);

                % Add everything that has no u in it  
                lnpsgz(k,:) = lnpsgz(k,:) - 1/2*sum(X'.*(XinvSig*X'),1) ...
                            - 1/2*sum(uu(:).*self.uinvSig{j}(:)) ...
                            + 1/2*log(det(self.uinvSig{j})) - D/2*log(2*pi) ...
                            + 1/2*logdetinvSigXX - self.nn{n}.NR/2*log(2*pi);
                        
                ugsz{k}.mu = mu;
                ugsz{k}.Sigma = Sigma;
                ugsz{k}.invSigma = invSigma;
                ugsz{k}.invSigmamu = invSigmamu;

            end

            psgz = exp(bsxfun(@minus,lnpsgz,max(lnpsgz)));
            psgz = bsxfun(@rdivide,psgz,sum(psgz,1));
            NA = sum(psgz,2);
            
            gb = self.nn{n}.gb{j}.mean;

            U = zeros(D,T);
            Y = zeros(1,T);
            for k=1:self.NLC
                Abar = self.AB{j}{k}.mean;
                Bbar = Abar(D+1);
                Abar = Abar(1:D);
                temp = bsxfun(@times,ugsz{k}.mu,psgz(k,:));
                U = U + temp;
                Y = Y + (gb(1)*(Abar'*ugsz{k}.mu+Bbar)+gb(2)).*psgz(k,:);
            end
                
        end
        
        function plot(self,n,j,Y,X)
            [Ypred,U,psgz] = self.getPredictions(n,j,X);
            MSE = mean((Y-Ypred').^2);
            T = size(X,1);
            [m,idx]=max(psgz);
            cc=jet(size(psgz,1));
            
            figure(n)
            scatter(U(1,:),Y,ones(1,T)*3,'k')
            hold on
            scatter(U(1,:),Ypred,ones(1,T)*3,cc(idx))
            title(['Nonlinearity ',num2str(j), ' with MSE ',num2str(MSE)])
            xlabel('U')
            ylabel('Y')
            
            hold off
            drawnow
        end
    end
end

   

