classdef PoiNLregressionWMatN < handle
    properties
        
        NR %number of regressors
        D  %dimension of non-linearity
        NC %number of clusters (max for DP approximation)
        NO %dimensions of the observations
        alpha 
        iswhite % are the regressors white?
        
        pi % prior probability of cluster assignment (dirichlet object)
        
        W % matrix normal wishart NR x D matix 
        
% One per cluster of these.
        u      % normal posterior on cluster mean
        Sigmau % cluster variance DxD.  Should probaby be 1/NC^D
        invSigmau
        logdetinvSigmau
        
        AB     % matrix normal wishart on parameters of nonlinearity
               % note that in the posterior the observation noise corresponds to U.  
               % This object is N0 x D+1, with (:,D+1) giving the
               % intercept.  

        % Suffstats
        SEu
        SEugz
        SEu1u1gz
        SEetau1gz
        SEetaetagz
        SEXu
        SEuu 
        SEXX
        SlnYfactorial

        lambdasave
        iters

% Latent Variables (no longer used
%       ugz    % latent means and covariances on u given z.   
               % ugz.mu{k}(:,t) is vector mean given cluster k 
               % ugz.Sigma{k} is covariance matrix.  Note the
               % independence on t.
               
%        p      % latent cluster assignments (NC x T) matrix
%        logptilde  % same as p sort of.
        NA     % number of assignments
        
% ELBO
        L
    end
    
    methods
        function self = PoiNLregressionWMatN(NR,NC,D,alpha,muY,chunkfactor)
            NO=1;
            self.iters=0;
            if(~exist('iswhite','var'))
                iswhite =2;
            end
            if(~exist('muY','var'))
                muY=1;
            end
            if(~exist('chunkfactor','var'))
                self.chunkfactor=1;
            else
                self.chunkfactor=chunkfactor; 
            end
            self.iswhite=iswhite;
            self.NR = NR;
            self.NO = NO;
            self.D = D;
            self.alpha = alpha;
            self.NC = NC;
            self.pi=dists.expfam.dirichlet(self.NC,alpha*ones(self.NC,1)/self.NC);
            
            if(iswhite==1)
%                self.W = dists.expfam.matrixnormalWishartwhiteOBS(NR,D,1/NR/muY,eye(D));
                self.W = dists.expfam.matrixnormalWishartwhiteOBS(NR,D,1/NR,eye(D));
            elseif(iswhite==2)
%                self.W = dists.expfam.matrixnormalWishartStaticXX(NR,D,1/NR/muY,eye(D));
                self.W = dists.expfam.matrixnormalWishartStaticXX(NR,D,1/NR,eye(D));
            else
%                self.W = dists.expfam.matrixnormalWishart(zeros(NR,D),eye(NR)/NR/muY,eye(D));    
%                self.W.mu = 1/sqrt(NR)*randn(NR,D);
                self.W = dists.expfam.matrixnormalWishart(zeros(NR,D),eye(NR)/NR,eye(D));    
            end
                      
%             self.Sigmau = eye(D);
%             self.invSigmau = eye(D);
%             self.logdetinvSigmau = D*log(1);            
%             self.Sigmau = (1/NC)^(2/D)*eye(D)/muY;
%             self.invSigmau = NC^(2/D)*eye(D)*muY;
%             self.logdetinvSigmau = 2*log(NC)+D*log(muY);            
             self.Sigmau = (1/NC)^(2/D)*eye(D);
             self.invSigmau = NC^(2/D)*eye(D);
             self.logdetinvSigmau = 2*log(NC);            
%             
            for k=1:self.NC
                self.u{k} = dists.expfam.MVN(zeros(D,1),eye(D));
                self.u{k}.mu_0 = randn(D,1);
                self.u{k}.mu=self.u{k}.mu_0;
%                self.u{k} = dists.expfam.MVN(zeros(D,1),eye(D)*muY);
%                self.u{k}.mu = randn(self.D,1)*sqrt(NC^(1/D));
%                self.u{k} = dists.expfam.MVN(zeros(D,1),eye(D));
%                if(D==1)
%                    self.u{k}.mu = k-self.NC/2;
%                    self.u{k}.mu_0 = k-self.NC/2;
%                    self.u{k}.mu = k/self.NC*4-2;
%                end
                self.AB{k} = dists.expfam.matrixnormalWishart([zeros(NO,D),log(muY)],eye(NO),eye(D+1));
                self.AB{k}.mu = randn(1,D+1)/4;
                self.AB{k}.mu(1,D+1)=log(muY);
            end
            self.L=-Inf;
            self.SEXX=[];
        end
        
        function DL = update(self,Y,X,niters,updateW)
            if(~exist('niters','var'))
                niters = 1;
                updateW = 1;
            elseif(~exist('updateW','var'))
                updateW=1;
            end
            for i=1:niters
                DL=self.L;
                self.iters=self.iters+1;
                self.update_suffstats(Y,X);
                self.updateparms(updateW);
                DL=self.L-DL;
                
                if(DL<0)
                   fprintf(['Warning: Lower bound decreased by ',num2str(abs(DL)/abs(self.L)*100),' percent at iter ',num2str(self.iters), '\n']) 
                end
%                fprintf([num2str(i/niters*100),' Percent Complete\n'])
            end
        end
        
        function [DL,p,logptilde,etaugz] = update_suffstats(self,Y,X,updateSEXX,p)
            if(~exist('updateSEXX','var'))
                updateSEXX = 0;
            end
            % Assumes that Y is Ns x NO and X is Ns x NR
            if(~exist('self.SlnYfactorial','var'))
                self.SlnYfactorial = sum(log(factorial(Y)));
            end
            if(updateSEXX==1)
                self.SlnYfactorial = sum(log(factorial(Y)));                
                self.chunkSEXX(X);
                if(self.iswhite==2)
                    self.W.setSEYY(self.SEXX);
                end                
            elseif(self.iswhite==2 && isempty(self.SEXX))
                self.chunkSEXX(X);
                self.W.setSEYY(self.SEXX);                
            elseif(self.iswhite==0 && isempty(self.SEXX))
                self.chunkSEXX(X);                
            end
            
            % In order to compute the cluster assignments we first compute
            % the approximate posterior over u given z 
            % Fortunately under the variational approximation all u(t,k)
            % have the same precision matrix for a given k.  Meanwhile, 
            % because of the conditional approximation, cluster assignments
            % only depende upon the u undependent terms, and the partition
            % function of the posterior on the u{k}'s.
 
            Ns=size(Y,1);
            logptilde = repmat(self.pi.loggeomean,1,Ns);            
            % Get prior on etagz and ugz
            
            % This bit only needed for computing lower bound correctly;
            logfactorialY=log(factorial(Y))';
            if(self.iswhite==1)  
                XWEinvUX = self.W.EtraceinvU;
            else 
                XWEinvUX = sum(sum(self.SEXX.*self.W.EinvU))/Ns;
            end
            logptilde = logptilde + repmat(1/2*self.W.ElogdetinvU - self.NR/2*log(2*pi) ...
                                           - logfactorialY -1/2*XWEinvUX,self.NC,1);
            % end this bit
            
            D=self.D;
            XTEXTinvU=(X*self.W.EXTinvU')';
            for k=1:self.NC
                abar = self.AB{k}.mean;
                bbar = abar(:,self.D+1);
                abar = abar(:,1:self.D);
                abm2 = self.AB{k}.EXTinvUX;
                
                etaugz{k}.invSigma = [self.AB{k}.invU.mean, ...
                    -self.AB{k}.invU.mean*abar; ...
                    -abar'*self.AB{k}.invU.mean, ...
                    abm2(1:self.D,1:self.D) + self.W.EXTinvUX + self.invSigmau]; 

                etaugz{k}.Sigma = inv(etaugz{k}.invSigma);
                etaugz{k}.Sigma2 = etaugz{k}.Sigma*[1,zeros(1,D);zeros(D,1),zeros(D,D);]*etaugz{k}.Sigma;                

                etaugz{k}.invSigmamu = repmat([self.AB{k}.invU.mean*bbar; ...
                                           -abm2(1:D,D+1) + self.invSigmau*self.u{k}.mean;],1,Ns);                
                etaugz{k}.invSigmamu(2:end,:) = etaugz{k}.invSigmamu(2:end,:) + XTEXTinvU;                  
                etaugz{k}.mu=etaugz{k}.Sigma*etaugz{k}.invSigmamu;   
                etaugz{k}.logZp = 1/2*sum(etaugz{k}.mu.*etaugz{k}.invSigmamu,1) ...
                                - 1/2*log(det(etaugz{k}.invSigma)) + (self.D+1)/2*log(2*pi);
                            
                lambda = dists.poissonnormal(etaugz{k}.mu(1,:)',repmat(1/etaugz{k}.Sigma(1,1),Ns,1));
%                 lambda = dists.poissonnormal(etaugz{k}.mu(1,:)',repmat(etaugz{k}.invSigma(1,1),Ns,1));
%                 temp = lambda.priormean;
%                 lambda.lambda(Y==0) = temp(Y==0);
%                 temp = (lambda.priormean+lambda.priorvariance).*Y./(Y+lambda.priorvariance);
%                 lambda.lambda(Y~=0) = temp(Y~=0);
                if(isempty(self.lambdasave))
                    self.lambdasave=repmat(mean(Y),[Ns,self.NC]);
                end
                lambda.lambda=self.lambdasave(:,k);
                lambda.updateSS(Y,1,5,1);
                self.lambdasave(:,k)=lambda.mean;
                
                [dExcoeff,dExxcoeff{k}] = lambda.effloglikelihood;                
                etaugz{k}.invSigmamu(1,:) = etaugz{k}.invSigmamu(1,:) + dExcoeff';

                etaugz{k}.mu = etaugz{k}.Sigma*etaugz{k}.invSigmamu ...
                             - bsxfun(@rdivide,etaugz{k}.Sigma2*etaugz{k}.invSigmamu,1./dExxcoeff{k}'+etaugz{k}.Sigma(1,1));

                logptilde(k,:) = logptilde(k,:) + etaugz{k}.logZp;

                %add everything that has no u in it.  Note this is missing
                %normalizers from p(x|u) and xxT terms. 
                
                logptilde(k,:) = logptilde(k,:) ...
                    + (Y.*lambda.eta.mu - lambda.mean - lambda.KLqpriorvec)'...
                    - 1/2*abm2(self.D+1,self.D+1) ...
                    + 1/2*self.AB{k}.invU.Elogdet - self.NO/2*log(2*pi) ...
                    - 1/2*trace(self.u{k}.secondmoment*self.invSigmau) - self.D/2*log(2*pi) ...
                    + 1/2*self.logdetinvSigmau;
            end

                % Ok so this gives logptilde(:,k) which must be converted
                % to an actual probability.  Note that there are missing
                % terms that are K independent associated with p(X|u)
                % that needs to be added in when computing lower bound...
            if(self.NC==1);
               p=ones(1,Ns);
               self.NA=Ns;
               L = 0;
               L = L + sum(logptilde(:));
            elseif(~exist('p','var'))
                p = exp(bsxfun(@minus,logptilde,max(logptilde)));
                p = bsxfun(@rdivide,p,sum(p,1));
                self.NA = sum(p,2);
                L = sum(sum(p.*(logptilde)));
                idx = find(p>0);            
                L = L - sum(p(idx).*log(p(idx)));
%                L = L - sum(log(p.^p));
            else
                self.NA = sum(p,2);
                L = sum(sum(p.*(logptilde)));
                idx = find(p>0);            
                L = L - sum(p(idx).*log(p(idx)));
%                L = L - sum(log(p.^p));
            end %if

            % As far as the u's and assignments are concerned the only
            % contribution to the lower bound comes from logptilde (this is
            % because we have done the equivalent of marginalizing out the
            % u's.  As is usually the case whem this is done, this is the
            % place where we are supposed to compute the lower bound.
            
            %%% here is where we updates the sufficient statistics
            self.SEu  = zeros(self.D,1);
            self.SEXu = zeros(self.NR,self.D);
            self.SEuu = zeros(self.D);
            for k=1:self.NC
                
                temp = bsxfun(@times,etaugz{k}.mu,p(k,:));                
                self.SEugz{k} = sum(temp(2:end,:),2);
                
                temp2 = etaugz{k}.mu*temp' + self.NA(k)*etaugz{k}.Sigma ...
                      - etaugz{k}.Sigma2*(p(k,:)*(1./(1./dExxcoeff{k}+etaugz{k}.Sigma(1,1))));            

                self.SEetaetagz{k} = temp2(1,1);
                self.SEu1u1gz{k} = [temp2(2:end,2:end),self.SEugz{k};self.SEugz{k}',self.NA(k);];
                
                self.SEetau1gz{k} = [temp2(1,2:end),sum(temp(1,:))];
                
                self.SEu = self.SEu + self.SEugz{k};
                self.SEXu = self.SEXu + (temp(2:end,:)*X)';
                self.SEuu = self.SEuu + self.SEu1u1gz{k}(1:self.D,1:self.D);
            end
            
%             if(self.iswhite==1)  
%                 L = L + Ns/2*self.W.ElogdetinvU - Ns*self.NR/2*log(2*pi) ...
%                                 - Ns/2*self.W.EtraceinvU ...
%                                 - self.SlnYfactorial;
%             elseif(self.iswhite==2)  
%                 L = L + Ns/2*self.W.ElogdetinvU - Ns*self.NR/2*log(2*pi) ...
%                                 - sum(sum(self.SEXX.*self.W.EinvU))/2 ...
%                                 - self.SlnYfactorial;
%             else 
%                 L = L ...
%                     - 1/2*sum(sum(self.SEXX.*self.W.EinvU)) ...
%                     + Ns/2*self.W.ElogdetinvU - Ns*self.NR/2*log(2*pi) ...
%                     - self.SlnYfactorial;
%             end
            if(self.NC>1)
                L = L - self.pi.KLqprior;
            end
            L = L - self.W.KLqprior;
            for k=1:self.NC
                L = L - self.AB{k}.KLqprior;
                L = L - self.u{k}.KLqprior;
            end
            DL = L-self.L;
            self.L=L;
            
        end
        
        function chunkSEXX(self,X)
            fprintf('Updating SEXX...')
            if(self.chunkfactor==1)
                self.SEXX=X'*X;
            else
                self.SEXX = zeros(size(X,2));
                chunksize=floor(size(X,1)/self.chunkfactor);
                for i=1:self.chunkfactor-1
                    self.SEXX = self.SEXX + X((i-1)*chunksize+1:i*chunksize,:)'*X((i-1)*chunksize+1:i*chunksize,:);
                end
                i=self.chunkfactor;
                self.SEXX = self.SEXX + X((i-1)*chunksize+1:end,:)'*X((i-1)*chunksize+1:end,:);
            end
            fprintf('done.\n')
        end
        
        function updateparms(self,updateW)
            if(~exist('updateW','var'))
                updateW=1;
            end
            self.pi.updateSS(self.NA);
            for k=1:self.NC
                self.u{k}.updateSS(self.invSigmau*self.SEugz{k}/self.NA(k),self.invSigmau,self.NA(k));
                self.AB{k}.updateSS(self.SEu1u1gz{k}/self.NA(k),self.SEetau1gz{k}/self.NA(k),self.SEetaetagz{k}/self.NA(k),self.NA(k));
            end
            N=sum(self.NA);
            if(updateW)
                self.W.updateSS(self.SEuu/N,self.SEXu/N,self.SEXX/N,N);
%             else
%                 W=self.W.mu;
%                 self.W.updateSS(self.SEuu/N,self.SEXu/N,self.SEXX/N,N);
%                 self.W.setmu(W);
            end
            
        end
        
        function W = init(self,Y,X,initW)
            if(~exist('initW','var'))
                initW=1;
            end
            W=(Y'*X)'/length(Y);
            W=W/sqrt(W'*W)*0.9; %is NRx1
            if(self.D>1)
                W(1,self.D)=0;
            end
            if(initW>0)
                for i=2:self.D
                    if(self.iswhite)
                        W(:,i) = 1/self.W.invU_0(1,1)*randn(self.NR,1)*self.W.V_0(1,1);
                        W(:,i) = randn(self.NR,1)*sqrt(var(W(:,1)))/10;
                    else
                        W(:,i) = self.W.invU.meaninv*randn(self.NR,1)*self.W.V_0(1,1);
                        W(:,i) = randn(self.NR,1)*sqrt(var(W(:,1)))/10;
                    end
                end
                self.W.mu=W;
                for k=1:self.NC
                    self.AB{k}.mu_0(self.D+1) = log(mean(Y));
                    self.AB{k}.mu(self.D+1) = log(mean(Y));
                end
                self.update(Y,X,initW,0)
            end
        end
        
        function fit(self,Y,X,iters,inititers,filliters)
            fprintf('Initializing...');    
            W=self.init(Y,X,inititers);
       
            for i=2:inititers
                self.update(Y,X,1);
            end
            
          %  self.update(Y,X,1);

            fprintf('done\n');
            figure(1);clf;hold on
            idx=randi(size(Y,1),size(Y,1),1);
            nho = ceil(size(Y,1)*0.1);
            
%             Yho=Y(idx(1:nho),:); Xho=X(idx(1:nho),:);
%             Y=Y(idx(nho+1:end),:); X=X(idx(nho+1:end),:);
            
            Lho = -Inf;
            LhoLast = -Inf;
            i=0;
%             while((i<iters & Lho>LhoLast) | i < iters/2)
            while( i<iters )
                i=i+1;
%                 LhoLast = Lho;
%                 Lho = self.update_suffstats(Yho,Xho);                
                self.update(Y,X,1);
%                 if(mod(i,5)==1 &  i < 50  &  filliters>0)
                if(i < 0.25*iters & filliters>0 & mod(i,4)==0)
                    self.fillunused(Y,X,0.10,filliters); 
                    self.update(Y,X,1);
                end
                if(mod(i,1)==0 && self.D>1)
                    for j=1:self.NO
                        figure(j+1)
                    for k=1:self.NC
                        temp = self.AB{k}.mean;
                        abar(k,:) = temp(j,:);
                        umu(k,:) = self.u{k}.mu';
                    end
                    subplot(3,1,1), bar(abar)
                    subplot(3,1,2), bar(self.W.mu)
                    subplot(3,1,3), bar(umu)
                    end
                    temp = self.W.EXTinvU';
                    for j=1:self.D
                        wcorr(j) = corr(temp(:,j),W(:,1));
                    end
%                     if(mod(i,10)==0)
%                         [mm,mmidx]=max(abs(wcorr));
%                         self.orthogonalizeW(mmidx)
%                     end


                    figure(1)
                    scatter(i,self.L)
                    MSE = self.plot(Y,X,self.NO+2);
                    fprintf('ELBO = %0.5g reached in %d iterations with Wcorr = %0.5g\n',self.L,i,max(abs(wcorr)));
                    self.NA'
                elseif(mod(i,1)==0)
                    figure(1)
                    scatter(i,self.L)
                    MSE = self.plot(Y,X,self.NO+2);%self.plot(Y(1:1000,1),X(1:1000,:),self.NO+2);
                    fprintf('ELBO = %0.5g reached in %d iterations with \n MSE = %0.5g and Wcorr = %0.5g\n',self.L,i,MSE,corr(self.W.EXTinvU',W));
                    self.NA'
                end
                drawnow
            end
            figure(1); hold off
        end

        function res = getfilter(self)    
             res = inv(self.W.EXTinvUX)*self.W.EXTinvU;
        end
                
        function fillunused(self,Y,X,pc,niters)
            idx=find(self.NA<1/2);
            if(length(idx)==0)
                fprintf('No Empty Clusters\n')
                return
            end
            [L,p,logptilde,etaugz] = update_suffstats(self,Y,X);
            NU = length(idx);
            Ns = size(Y,1);
            Ns = ceil(Ns*pc);
            fprintf(['Filling ',num2str(NU),' Clusters\n'])
            
            if(NU>0)
%                NU=min(NU,2);
                m = max(logptilde);
               [err,idxe]=sort(m);

                modeltemp = PoiNLregression(self.NR,NU,self.D,self.alpha,self.iswhite,mean(Y));
                if(NU>1)
                    [z,c]=kmeans([X(idxe(1:Ns),:)*self.getfilter()',Y(idxe(1:Ns))],NU);
                    c=c(:,1:end-1);
                else
                    c=mean(X(idxe(1:Ns),:)*self.getfilter()');
                    z=ones(Ns,1);
                end
                for k=1:NU
                    modeltemp.AB{k}.mu(1,self.D+1)=log(mean(Y(idxe(1:Ns))));
                    modeltemp.AB{k}.mu(1,1:self.D)=zeros(1,self.D);
                    NA=sum(z==k);
                    if(NA>0)                        
                        modeltemp.u{k}.mu = c(k,:)';
                        modeltemp.u{k}.mu_0 = c(k,:)';
                        modeltemp.u{k}.Sigma = eye(self.D)/NA;
                        modeltemp.u{k}.invSigma = eye(self.D)*NA;
                        modeltemp.u{k}.invSigmamu = c(k,:)'*NA;
                        modeltemp.u{k}.setUpdated(false);                    
                    end
                end
%                modeltemp.initialize(Y(idxe(1:Ns),:),X(idxe(1:Ns),:),1);
                for i=1:niters
                    modeltemp.W = self.W.copy;
                    modeltemp.update(Y(idxe(1:Ns),:),X(idxe(1:Ns),:),1);
                end
                for i=1:NU
                    self.AB{idx(i)}=modeltemp.AB{i};
                    self.u{idx(i)}=modeltemp.u{i};
                end
                self.pi.alpha=self.pi.alpha_0;
            end
        end
                  
        function [Ypred,Upred,p] = getPredictions(self,Y,X)
            
%  To really compute this correctly we need to set the assigments without
%  using Y.
            Ns=size(Y,1);
            logptilde = repmat(self.pi.loggeomean,1,Ns);
            XTEXTinvU=(X*self.W.EXTinvU')';
            KLqprior=self.KLqprior;
            % Get prior on etagz and ugz given X
            D=self.D;
            for k=1:self.NC
                abar = self.AB{k}.mean;
                bbar = abar(:,self.D+1);
                abar = abar(:,1:self.D);
                abm2 = self.AB{k}.EXTinvUX;
                
                etaugz{k}.invSigma = [self.AB{k}.invU.mean, ...
                    -self.AB{k}.invU.mean*abar; ...
                    -abar'*self.AB{k}.invU.mean, ...
                    abm2(1:self.D,1:self.D) + self.W.EXTinvUX + self.invSigmau]; 

                etaugz{k}.Sigma = inv(etaugz{k}.invSigma);
                etaugz{k}.Sigma2 = etaugz{k}.Sigma*[1,zeros(1,D);zeros(D,1),zeros(D,D);]*etaugz{k}.Sigma;                

                etaugz{k}.invSigmamu = repmat([self.AB{k}.invU.mean*bbar; ...
                                           -abm2(1:D,D+1) + self.invSigmau*self.u{k}.mean;],1,Ns);                
                etaugz{k}.invSigmamu = etaugz{k}.invSigmamu + [zeros(self.NO,Ns);XTEXTinvU;];                  
                etaugz{k}.mu=etaugz{k}.Sigma*etaugz{k}.invSigmamu;   
                etaugz{k}.logZp = 1/2*sum(etaugz{k}.mu.*etaugz{k}.invSigmamu,1) ...
                                - 1/2*log(det(etaugz{k}.invSigma)) + (self.D+1)/2*log(2*pi);

                %add logZ
                logptilde(k,:) = logptilde(k,:) + etaugz{k}.logZp;

                %add everything that has no ugz in it.
                logptilde(k,:) = logptilde(k,:) ...
                    - 1/2*abm2(self.D+1,self.D+1) ...
                    + 1/2*self.AB{k}.invU.Elogdet - self.NO/2*log(2*pi) ...
                    - 1/2*trace(self.u{k}.secondmoment*self.invSigmau) - self.D/2*log(2*pi) ...
                    + 1/2*self.logdetinvSigmau;
            end

                % Ok so this gives logptilde(:,k) which must be converted
                % to an actual probability.  Note that there are missing
                % terms that are K independent associated with p(X|u)
                % that needs to be added in when computing lower bound...
            if(self.NC==1);
               p=ones(1,Ns);
            elseif(~exist('p','var'))
                p = exp(bsxfun(@minus,logptilde,max(logptilde)));
                p = bsxfun(@rdivide,p,sum(p,1));
            end
            
            Upred.mu = zeros(self.D,Ns);
            Ypred.mu = zeros(1,Ns);
            for k=1:self.NC
                Upred.mu = Upred.mu + bsxfun(@times,etaugz{k}.mu(2:end,:),p(k,:));
                Ypred.mu = Ypred.mu + bsxfun(@times,exp(etaugz{k}.mu(1,:)+1/2*etaugz{k}.Sigma(1,1)),p(k,:));
            end
            
        end
        
        function [Evar,EDev,KLqprior] = plot(self,Y,X,fignum)
            D=self.D;
            [Ypred,Upred,p] = self.getPredictions(Y,X);
            
            MSE = mean(mean((Y-Ypred.mu').^2));
            Evar = 1-MSE/var(Y);
            
            muY = mean(Y);
            idxD=find(Y>0);
            satDev = -2*(sum(Y(idxD).*log(Y(idxD))) - sum(Y));
            Dev = -2*(sum(Y.*log(Ypred.mu')) - sum(Ypred.mu'));
            nullDev =  -2*(sum(Y)*log(muY) - muY*size(Y,1));

            deltaDev = nullDev - Dev;
            nullDev = nullDev - satDev;
            Dev = Dev - satDev;
            EDev=Dev/size(X,1);

            if(self.NC==1)
                idx=ones(1,size(X,1));
            else
                [m,idx]=max(p);
            end
            
            cc=jet(self.NC);
            for i=1:self.NO
                figure(fignum+i-1)
                clf
                if(self.D==1)
                    [mm,bins]=hist(Upred.mu(1,:),41);
                    for j=2:40
                        idx5 = Upred.mu(1,:)>bins(j-1) & Upred.mu(1,:)<bins(j+1);
                        u(j-1)=mean(Upred.mu(1,idx5));
                        fu(j-1)=mean(Y(idx5));
                    end
                    scatter(Upred.mu(1,:),Y(:,i)',3*ones(size(Ypred.mu(i,:))),'k.')            
                    hold on
                    scatter(Upred.mu(1,:),Ypred.mu(i,:),9*ones(size(Ypred.mu(i,:))),cc(idx,:),'.')
                    plot(u,fu,'k')
                    hold off
                else
                    scatter3(Upred.mu(1,:),Upred.mu(2,:),Y(:,i)',9*ones(size(Ypred.mu(i,:))),'k.')            
                    hold on
                    scatter3(Upred.mu(1,:),Upred.mu(2,:),Ypred.mu(i,:),9*ones(size(Ypred.mu(i,:))),cc(idx,:),'.')
                    hold off
                    view(0,0)
                end
            end
            if(self.D>1)
                figstart=fignum+self.NO;
                np=3;
                mu1 = mean(Upred.mu(2,:));
                std1 = std(Upred.mu(2,:));
                u1=linspace(mu1-2*std1,mu1+2*std1,np^2+1);

                [temp,idx2]=histc(Upred.mu(2,:),u1);

                for i=1:self.NO
                    figure(figstart+i-1)
                    k=0;
                    for j=1:np
                    for jj=1:np
                        k=k+1;
                        idx3=find(idx2==k);  
                        if(length(idx3)>0)
                           subplot(np,np,k), scatter(Upred.mu(1,idx3),Y(idx3,i)','k.');
                           hold on; scatter(Upred.mu(1,idx3),Ypred.mu(i,idx3),10*ones(1,length(idx3)),cc(idx(idx3),:),'.')
                           hold off;
                           axtemp=axis;
                           axtemp(3)=0;
                           axtemp(4)=2;
                           axis(axtemp);
                        end
                    end
                    end
                    drawnow
                end
            end
            drawnow
        end
        
        function flippolarityA(self,Aidx) % here Aidx is a receptive field index.
            % this function rotates A associated with the dominant cluster
            % so that AB{k}.mu(:,j)=0 for all k and j~=Aidx or j==self.D+1;
            [m,idx]=max(self.NA);
            
            temp=self.AB{idx}.mu; %is 1 x D
            P=eye(self.D);
            P(Aidx,Aidx)=-1;
            % P is now a DxD 

            invP=inv(P);
            
            self.Sigmau = invP*self.Sigmau*invP';
            self.invSigmau = P'*self.invSigmau*P;

            for k=1:self.NC
                self.u{k}.mu = invP*self.u{k}.mu;
                self.u{k}.invSigma = P'*(self.u{k}.invSigma-self.u{k}.invSigma_0)*P+self.u{k}.invSigma_0;
%                self.u{k}.invSigma = P'*(self.u{k}.invSigma)*P;
                self.u{k}.Sigma = inv(self.u{k}.invSigma);
                self.u{k}.invSigmamu = self.u{k}.invSigma*self.u{k}.mu;
                self.u{k}.setUpdated(false);
            end
            self.W.mu = self.W.mu*P;
%            self.W.invV = invP*self.W.invV*invP';
            self.W.invV = invP*(self.W.invV-self.W.invV_0)*invP'+self.W.invV_0;
            self.W.V = inv(self.W.invV);
            if(self.iswhite)
                self.W.muTmu = P'*self.W.muTmu*P;
                self.W.EXTinvUX = P'*self.W.EXTinvUX*P;
                self.W.EXTinvU = P'*self.W.EXTinvU;            
            end
            
            P(self.D+1,self.D+1)=1;
            invP=inv(P);
            for k=1:self.NC
                self.AB{k}.mu = self.AB{k}.mu*P;
                self.AB{k}.invV = invP*(self.AB{k}.invV-self.AB{k}.invV_0)*invP'+self.AB{k}.invV_0;
%                self.AB{k}.invV = invP*(self.AB{k}.invV)*invP';
                self.AB{k}.V=inv(self.AB{k}.invV);
            end
            
        end        
        
        function rotateA(self,Aidx) % here Aidx is a receptive field index
            % this function rotates A associated with the dominant cluster
            % so that AB{k}.mu(:,j)=0 for all k and j~=Aidx or j==self.D+1;
            
            Abar = self.AB{idx}.mu(1:self.D);            
            [m,idx]=max(self.NA);
            
            Abar = zeros(1,self.D);
            for k=1:self.NC
                Abar = Abar + self.AB{k}.mu(1,1:self.D)*self.NA(k);
            end
            Abar = Abar/sum(self.NA);
            P=eye(self.D);
            for k=1:Aidx-1
                vec1 = Abar(1,Aidx);
                vec2 = Abar(1,k);
                theta=-angle(vec1+sqrt(-1)*vec2);
                R1=eye(self.D);
                R1(Aidx,Aidx)=cos(theta);
                R1(Aidx,k)=-sin(theta);
                R1(k,Aidx)=sin(theta);
                R1(k,k)=cos(theta);
                P=P*R1;
            end
            for k=Aidx+1:self.D
                vec1 = Abar(1,Aidx);
                vec2 = Abar(1,k);
                theta=-angle(vec1+sqrt(-1)*vec2);
                R1=eye(self.D);
                R1(Aidx,Aidx)=cos(theta);
                R1(Aidx,k)=sin(theta);
                R1(k,Aidx)=-sin(theta);
                R1(k,k)=cos(theta);
                P=P*R1;                
            end
                            
            % P is now a DxD 

            invP=inv(P);
            
            self.Sigmau = invP*self.Sigmau*invP';
            self.invSigmau = P'*self.invSigmau*P;

            for k=1:self.NC
                self.u{k}.mu = invP*self.u{k}.mu;
                self.u{k}.invSigma = P'*(self.u{k}.invSigma-self.u{k}.invSigma_0)*P+self.u{k}.invSigma_0;
%                self.u{k}.invSigma = P'*(self.u{k}.invSigma)*P;
                self.u{k}.Sigma = inv(self.u{k}.invSigma);
                self.u{k}.invSigmamu = self.u{k}.invSigma*self.u{k}.mu;
                self.u{k}.setUpdated(false);
            end
            self.W.mu = self.W.mu*P;
            self.W.invV = invP*self.W.invV*invP';
%            self.W.invV = invP*(self.W.invV-self.W.invV_0)*invP'+self.W.invV_0;
            self.W.V = inv(self.W.invV);
            if(self.iswhite)
                self.W.muTmu = P'*self.W.muTmu*P;
                self.W.EXTinvUX = P'*self.W.EXTinvUX*P;
                self.W.EXTinvU = P'*self.W.EXTinvU;            
            end
            
            P(self.D+1,self.D+1)=1;
            invP=inv(P);
            for k=1:self.NC
                self.AB{k}.mu = self.AB{k}.mu*P;
%                self.AB{k}.invV = invP*(self.AB{k}.invV-self.AB{k}.invV_0)*invP'+self.AB{k}.invV_0;
                self.AB{k}.invV = invP*(self.AB{k}.invV)*invP';
                self.AB{k}.V=inv(self.AB{k}.invV);
            end
            
        end
 
        function rotateU(self,Y,X) %only for 2D nonlinearities
            % Finds the rotation on u that maximizes the correlations between
            % u and Y
            if(self.D~=2)
                fprintf('D ~=2 doing nothing\n')
                return
            end
            
            [Ypred,Upred]=self.getPredictions(Y,X);
            
            [A,B]=canoncorr(log(Ypred.mu'),Upred.mu');
            B=B'/norm(B);
            B(2,1)=-B(1,2);
            B(2,2)=B(1,1);
                        
            P=inv(B);                
            invP=B;
            
            self.Sigmau = invP*self.Sigmau*invP';
            self.invSigmau = P'*self.invSigmau*P;

            for k=1:self.NC
                self.u{k}.mu = invP*self.u{k}.mu;
                self.u{k}.invSigma = P'*(self.u{k}.invSigma-self.u{k}.invSigma_0)*P+self.u{k}.invSigma_0;
%                self.u{k}.invSigma = P'*(self.u{k}.invSigma)*P;
                self.u{k}.Sigma = inv(self.u{k}.invSigma);
                self.u{k}.invSigmamu = self.u{k}.invSigma*self.u{k}.mu;
                self.u{k}.setUpdated(false);
            end
            self.W.mu = self.W.mu*P;
            self.W.invV = invP*self.W.invV*invP';
%            self.W.invV = invP*(self.W.invV-self.W.invV_0)*invP'+self.W.invV_0;
            self.W.V = inv(self.W.invV);
            if(self.iswhite)
                self.W.muTmu = P'*self.W.muTmu*P;
                self.W.EXTinvUX = P'*self.W.EXTinvUX*P;
                self.W.EXTinvU = P'*self.W.EXTinvU;            
            end
            
            P(self.D+1,self.D+1)=1;
            invP=inv(P);
            for k=1:self.NC
                self.AB{k}.mu = self.AB{k}.mu*P;
%                self.AB{k}.invV = invP*(self.AB{k}.invV-self.AB{k}.invV_0)*invP'+self.AB{k}.invV_0;
                self.AB{k}.invV = invP*(self.AB{k}.invV)*invP';
                self.AB{k}.V=inv(self.AB{k}.invV);
            end
            
        end
        
        function projectontoW(self,W,Nsf) % here W is the linear receptive field.
            % this function rotates A so that the effective filter applied to get u(1)
            % is as close to W as possible 
            %%%%%%% ONLY FOR D=2;
            
            if(self.D~=2)
                fprintf('only valid for D==2\n')
                return
            end
            if(~exist('Nsf','var'))
                Nsf=0;
            end
            temp=self.getfilter();
            W=W(1,1:end-Nsf);
            temp=temp(:,1:end-Nsf);
            d10 = (temp(1,:)*W');
            d20 = (temp(2,:)*W');
            
            if(abs(d20)<1e-14)
                fprintf('already done\n')
                return
            end
            
            a=1/sqrt(1+(d20/d10)^2)*sign(d10);
            b=d20/d10*a;
            
            P(1,1)=a;
            P(1,2)=b;
            P(2,1)=-b;
            P(2,2)=a;

            P=P';
            
            invP=inv(P);
%            P=inv(invP);
            
            self.Sigmau = invP*self.Sigmau*invP';
            self.invSigmau = P'*self.invSigmau*P;

            for k=1:self.NC
                self.u{k}.mu = invP*self.u{k}.mu;
                self.u{k}.invSigma = P'*(self.u{k}.invSigma-self.u{k}.invSigma_0)*P+self.u{k}.invSigma_0;
%                self.u{k}.invSigma = P'*(self.u{k}.invSigma)*P;
                self.u{k}.Sigma = inv(self.u{k}.invSigma);
                self.u{k}.invSigmamu = self.u{k}.invSigma*self.u{k}.mu;
                self.u{k}.setUpdated(false);
            end
            self.W.mu = self.W.mu*P;
            self.W.invV = invP*self.W.invV*invP';
%            self.W.invV = invP*(self.W.invV-self.W.invV_0)*invP'+self.W.invV_0;
            self.W.V = inv(self.W.invV);
            if(self.iswhite)
                self.W.muTmu = P'*self.W.muTmu*P;
                self.W.EXTinvUX = P'*self.W.EXTinvUX*P;
                self.W.EXTinvU = P'*self.W.EXTinvU;            
            end
            
            P(self.D+1,self.D+1)=1;
            invP=inv(P);
            for k=1:self.NC
                self.AB{k}.mu = self.AB{k}.mu*P;
%                self.AB{k}.invV = invP*(self.AB{k}.invV-self.AB{k}.invV_0)*invP'+self.AB{k}.invV_0;
                self.AB{k}.invV = invP*(self.AB{k}.invV)*invP';
                self.AB{k}.V=inv(self.AB{k}.invV);
            end

            
        end
        
        
        function orthogonalizeW(self,Widx) % here idx is the linear receptive field.
%            temp=self.W.mu; % is NR X D
            temp=self.getfilter; %is D x NR
            [m,idx]=sort(-diag(temp*temp'));
            t=find(idx==Widx);
            idx(2:t)=idx(1:t-1);
            idx(1)=Widx;
            P=zeros(self.D,self.D);
            for i=1:self.D
                P(i,idx(i))=1;
            end

            temp=P*temp;
            
            for i=1:self.D
                A=eye(self.D);
                for j=1:i-1
                    A(i,j) = -temp(i,:)*temp(j,:)'/(temp(j,:)*temp(j,:)'); 
                end
                P=A*P;
                temp=A*temp;
            end
            
            % P is now a DxD matrix that moves the linear receptive field
            % to the first dimension of u and orthogonalizes the rest.

            invP=P;
            P=inv(P);
            
            self.Sigmau = invP*self.Sigmau*invP';
            self.invSigmau = P'*self.invSigmau*P;

            for k=1:self.NC
                self.u{k}.mu = invP*self.u{k}.mu;
                self.u{k}.invSigma = P'*(self.u{k}.invSigma-self.u{k}.invSigma_0)*P+self.u{k}.invSigma_0;
%                self.u{k}.invSigma = P'*(self.u{k}.invSigma)*P;
                self.u{k}.Sigma = inv(self.u{k}.invSigma);
                self.u{k}.invSigmamu = self.u{k}.invSigma*self.u{k}.mu;
                self.u{k}.setUpdated(false);
            end
            self.W.mu = self.W.mu*P;
%            self.W.invV = invP*self.W.invV*invP';
            self.W.invV = invP*(self.W.invV-self.W.invV_0)*invP'+self.W.invV_0;
            self.W.V = inv(self.W.invV);
            if(self.iswhite)
                self.W.muTmu = P'*self.W.muTmu*P;
                self.W.EXTinvUX = P'*self.W.EXTinvUX*P;
                self.W.EXTinvU = P'*self.W.EXTinvU;            
            end
            
            P(self.D+1,self.D+1)=1;
            invP=inv(P);
            for k=1:self.NC
                self.AB{k}.mu = self.AB{k}.mu*P;
                self.AB{k}.invV = invP*(self.AB{k}.invV-self.AB{k}.invV_0)*invP'+self.AB{k}.invV_0;
%                self.AB{k}.invV = invP*(self.AB{k}.invV)*invP';
                self.AB{k}.V=inv(self.AB{k}.invV);
            end
            
        end
        
        function res = KLqprior(self)
            res = self.pi.KLqprior + self.W.KLqprior;
            for k=1:self.NC
                res = res + self.u{k}.KLqprior + self.AB{k}.KLqprior;
            end
        end
        
    end
   
end
