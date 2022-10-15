classdef NLregression < handle
    
    %Bayesian implementation of a non-linear regression with explicit
    %dimensionality reduction, i.e. models y = f(W*x) with a piecewise
    %linear approximtion to f.  We constrain the dimensionality/complexity
    %of f by fixing the dimensionality D of u=Wx.  I
    %
    % Generative model: 
    % For each pair of observations Y(t,:), X(t,:) 
    %     Sample a latent cluster assignment z(t)
    %     Sample a low dimensional latent u(t)|z(t)
    %     Sample regressors X(t,:) =  u(t)*W
    %     Sample prediction y(t,:) = y(t)*A_z(t) + b_z(t)
    %
    %The advantage here is that conditioned on latent assignment we can
    %do a fully Bayesian linear regresssion in a single VB update
    %ensuring rapid convergence. 
    
    properties
        
        NR %number of regressors
        D  %dimension of non-linearity
        NC %number of clusters (max for DP approximation)
        NO %dimensions of the observations
        alpha
        iswhite
        
        pi % prior probability of cluster assignment (dirichlet object)
        
        W % matrix normal wishart NR x D matix 
        
% One per cluster of these.
        u      % normal posterior on cluster mean
        Sigmau % cluster variance DxD.  Should probaby be 1/NC^D
        invSigmau
        logdetinvSigmau
        
        AB     % matrix normal wishart on parameters of nonlinearity
               % note that in the posterior the obsrvation noise corresponds to U.  
               % This object is N0 x D+1, with (:,D+1) giving the
               % intercept.  
        % Suffstats
        SEu
        SEugz
        SEu1u1gz
        SEYu1gz
        SEYYgz
        SEXu
        SEuu 
        SEXX 
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
        iters
    end
    
    methods
        function self = NLregression(NO,NR,D,NC,alpha,iswhite)
            if(~exist('iswhite','var'))
                iswhite =2;
            end
            self.iters=0;
            self.iswhite=iswhite;
            self.NR = NR;
            self.NO = NO;
            self.D = D;
            self.alpha = alpha;
            self.NC = NC;
            self.pi=dists.expfam.dirichlet(NC,alpha*ones(NC,1)/NC);
            
            if(iswhite==1)
                self.W = dists.expfam.matrixnormalWishartwhiteOBS(NR,D,1,eye(D));
            elseif(iswhite==2)
                self.W = dists.expfam.matrixnormalWishartStaticXX(NR,D,1/NR,eye(D));
            else
                self.W = dists.expfam.matrixnormalWishart(zeros(NR,D),eye(NR)/NR,eye(D));
%                self.W = dists.expfam.matrixnormalWishart(zeros(NR,D),eye(NR),eye(D));
            end
            
            self.Sigmau = 1/NC^(2/D)*eye(D);
            self.invSigmau = NC^(2/D)*eye(D);
            self.logdetinvSigmau = 2*log(NC);
%             self.Sigmau = eye(D);
%             self.invSigmau = eye(D);
%             self.logdetinvSigmau = D*log(1);
            
            
            for k=1:self.NC
                self.u{k} = dists.expfam.MVN(zeros(D,1),eye(D));
                %self.u{k}.mu_0 = randn(D,1);
                self.u{k}.mu = randn(D,1);%self.u{k}.mu_0;
                self.AB{k} = dists.expfam.matrixnormalWishart(zeros(NO,D+1),eye(NO),eye(D+1));
            end
            
            self.L=-Inf;
        end
        
        function [DL, i] = update(self,Y,X,niters,updateSEXX)
            if(~exist('niters','var'))
                niters = 1;
                updateSEXX=0;
            elseif(~exist('updateSEXX','var'))
                updateSEXX=0;
            end
            for i=1:niters
                DL = self.update_suffstats(Y,X,updateSEXX);
                updateSEXX=0;
                self.updateparms;
                self.iters=self.iters+1;
                if(DL<0)
                    fprintf('Lower bound decreasing at iteration %d\n', self.iters)
                    if abs(DL/self.L) < 1E-9
                        break
                    end
                end
%                fprintf('Iteration %d, lower bound %f\n', i, self.L)
            end
        end
        
        function [DL,p,logptilde,ugz] = update_suffstats(self,Y,X,updateSEXX,p)
            
            if(~exist('updateSEXX','var'))
                updateSEXX = 0;
            end
            if(isempty(self.SEXX))
                updateSEXX = 1;
            end
            % Assumes that Y is Ns x NO and X is Ns x NR
            
            % In order to compute the cluster assignments we first compute
            % the approximate posterior over u given z 
            % Fortunately under the variational approximation all u(t,k)
            % have the same precision matrix for a given k.  Meanwhile, 
            % because of the conditional approximation, cluster assignments
            % only depende upon the u undependent terms, and the partition
            % function of the posterior on the u{k}'s.
            DL=self.L;
            Ns=size(Y,1);
            logptilde = repmat(self.pi.loggeomean,1,Ns);

            WXprime = (X*self.W.EXTinvU')';
            for k=1:self.NC
                abar = self.AB{k}.mean;
                bbar = abar(:,self.D+1);
                abar = abar(:,1:self.D);
                yminusb = bsxfun(@minus,Y',bbar);

                ugz{k}.invSigma = abar'*self.AB{k}.invU.mean*abar ...
                    + self.NO*self.AB{k}.V(1:self.D,1:self.D) ...
                    + self.W.EXTinvUX + self.invSigmau; 

                ugz{k}.Sigma = inv(ugz{k}.invSigma);

                ugz{k}.invSigmamu = abar(:,1:self.D)'*self.AB{k}.invU.mean*yminusb;
                ugz{k}.invSigmamu = bsxfun(@minus,ugz{k}.invSigmamu,self.NO*self.AB{k}.V(1:self.D,self.D+1)) ...
                    + WXprime;
                ugz{k}.invSigmamu = bsxfun(@plus,ugz{k}.invSigmamu,self.invSigmau*self.u{k}.mean);
                ugz{k}.mu=ugz{k}.Sigma*ugz{k}.invSigmamu;   

%                ugz{k}.logZ = 1/2*sum(ugz{k}.mu.*ugz{k}.invSigmamu,1) ...
%                    + 1/2*log(det(ugz{k}.Sigma)) + self.D/2*log(2*pi);

                %add logZ
                logptilde(k,:) = logptilde(k,:) + 1/2*sum(ugz{k}.mu.*ugz{k}.invSigmamu,1) ...
                    + 1/2*log(det(ugz{k}.Sigma)) + self.D/2*log(2*pi);

                %add everything that has no ugz in it.
                logptilde(k,:) = logptilde(k,:) ...
                    - 1/2*sum(self.AB{k}.invU.mean*(yminusb).*yminusb,1) ...
                    - 1/2*self.NO*self.AB{k}.V(self.D+1,self.D+1) ...
                    + 1/2*self.AB{k}.invU.Elogdet - self.NO/2*log(2*pi) ...
                    - 1/2*trace(self.u{k}.secondmoment*self.invSigmau) - self.D/2*log(2*pi) ...
                    + 1/2*self.logdetinvSigmau;
            end

                % Ok so this gives logptilde(:,k) which must be converted
                % to an actual probability.
            if(self.NC==1);
               p=ones(1,Ns);
               self.NA(1)=Ns;
               self.L = 0;
               self.L = self.L + sum(logptilde(:));
            elseif(~exist('p','var'))
                p = exp(bsxfun(@minus,logptilde,max(logptilde)));
                p = bsxfun(@rdivide,p,sum(p,1));
                self.NA = sum(p,2);
                self.L = -self.pi.KLqprior;
                self.L = self.L + sum(sum(p.*(logptilde)));
                idx = find(p>0);            
                self.L = self.L - sum(p(idx).*log(p(idx)));
            else
                self.NA = sum(p,2);
                self.L = sum(logptilde(:));
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
                
                temp = bsxfun(@times,ugz{k}.mu,p(k,:));

                self.SEugz{k} = sum(temp,2);
                self.SEu = self.SEu + self.SEugz{k};
                self.SEu1u1gz{k} = [ugz{k}.mu*temp' + self.NA(k)*ugz{k}.Sigma,self.SEugz{k}; ...
                                     self.SEugz{k}',self.NA(k);];
                
                self.SEYu1gz{k} = [(temp*Y)',(p(k,:)*Y)'];
                self.SEYYgz{k} = bsxfun(@times,Y',p(k,:))*Y;
                self.SEXu = self.SEXu + (temp*X)';
                self.SEuu = self.SEuu + self.SEu1u1gz{k}(1:self.D,1:self.D);
            end
            
            if(self.iswhite==1)
                self.SEXX=[]; 
            elseif(isempty(self.SEXX))
                self.SEXX = X'*X;
                if(self.iswhite==2)
                    self.W.setSEYY(self.SEXX);                
                end
            elseif(updateSEXX)
                self.SEXX = X'*X;
                if(self.iswhite==2)                 
                    self.W.setSEYY(self.SEXX);                
                end
            end
            
            if(self.iswhite==1)  
                self.L = self.L + Ns/2*self.W.ElogdetinvU - Ns*self.NR/2*log(2*pi) ...
                                - Ns/2*self.W.EtraceinvU;
            elseif(self.iswhite==2)
                self.L = self.L + Ns/2*self.W.ElogdetinvU - Ns*self.NR/2*log(2*pi) ...
                    - sum(sum(self.SEXX.*self.W.EinvU))/2;
            else
                self.L = self.L ...
                    + Ns/2*self.W.ElogdetinvU - Ns*self.NR/2*log(2*pi) ...
                    - 1/2*sum(sum(self.SEXX.*self.W.EinvU));
            end    
            self.L = self.L - self.W.KLqprior;
            for k=1:self.NC
                self.L = self.L - self.AB{k}.KLqprior;
                self.L = self.L - self.u{k}.KLqprior;
            end
            DL=self.L-DL;

%            if(DL<0)
%                fprintf(['Warning Lower Bound Decreasing (DL/L = ',num2str(DL/abs(self.L)),') at iteration ',num2str(self.iters),'\n'])
%            end
        end

        function updateparms(self,updateW)
            
            if(~exist('updateW','var'))
                updateW=1;
            end
            self.pi.updateSS(self.NA);
            for k=1:self.NC
                self.u{k}.updateSS(self.invSigmau*self.SEugz{k}/self.NA(k),self.invSigmau,self.NA(k));
                self.AB{k}.updateSS(self.SEu1u1gz{k}/self.NA(k),self.SEYu1gz{k}/self.NA(k),self.SEYYgz{k}/self.NA(k),self.NA(k));
            end
            N=sum(self.NA);
            if(updateW)
                self.W.updateSS(self.SEuu/N,self.SEXu/N,self.SEXX/N,N);
            end
        end
        
        function initialize(self,Y,X,iters,Ns)
            muX=mean(X);
            muY=mean(Y);
            X=bsxfun(@plus,X,-muX);
            Y=bsxfun(@plus,Y,-muY);
            C=X'*X/size(X,1);
            W=inv(C+eye(size(C))/size(C,1)*mean(diag(C)))*(Y'*X)'/size(X,1);
            X=bsxfun(@plus,X,muX);
            Y=bsxfun(@plus,Y,muY);
            if(self.NO<=self.D)
                self.W.mu(:,1:self.NO)=W(1:self.NO,:)'/self.NR;
            else
                self.W.mu(:,1:self.D)=W(1:self.D,:)'/self.NR;
            end
            
            if(~exist('Ns','var'))
                temp=[Y,X];
                Ns=size(temp,1);
                idx=[1:Ns];
            else
                idx = randperm(size(X,1));
                idx = idx(1:Ns);
                temp = [Y(idx,:),X(idx,:)];
            end
            
            psave=zeros(self.NC,Ns);
            z=kmeans(temp,self.NC);
            NAsave = hist(z,[1:self.NC]);
            psave = (ones(Ns,1)*[1:self.NC] == z*ones(1,self.NC));
            for i=1:iters
                self.update_suffstats(Y(idx,:),X(idx,:),1,psave');
                self.updateparms(0);
            end
        end
        
        function fit(self,Y,X,iters,inititers,fill)
            fprintf('Initializing...');    
            if(inititers>0)
                self.initialize(Y,X,inititers);
            end
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
                if(i < 0.35*iters & fill>0 & mod(i,3)==0)
                    self.fillunused(Y,X,0.10); 
                    self.update(Y,X,1);
                end
                if(mod(i,10)==0 && self.D>1)
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
                else
                    figure(1)
                    scatter(i,self.L)
                    MSE = self.plot(Y,X,2);
                    fprintf('ELBO = %0.5g reached in %d iterations with MSE = %0.5g\n',self.L,i,MSE);
                end
                drawnow
            end
            figure(1); hold off
        end

        function res = getfilter(self)    
             res = inv(self.W.EXTinvUX)*self.W.EXTinvU;
        end
                
        function fillunused(self,Y,X,pc)
            idx=find(self.NA<1/2);
            NU = length(idx);
            if(NU==0)
                fprintf('No Empty Clusters\n')
                return
            end
            fprintf(['Filling ',num2str(NU),' Clusters\n'])

            [L,p,logptilde,ugz] = update_suffstats(self,Y,X,0);
            Ns = size(Y,1);
            Ns = ceil(Ns*pc);

%                 mx = max(logptilde);
%                [err,idxe]=sort(mx);
               [Ypred,Upred]=self.getPredictions(Y,X);
               [err,idxe]=sort((Ypred.mu-Y').^2,'descend');

            modeltemp = NLregression(self.NO,self.NR,self.D,NU,self.alpha,self.iswhite);
            modeltemp.W = self.W;
%            z = kmeans([Y(idxe(1:Ns),:),X(idxe(1:Ns),:)],NU);
%            utemp = X(idxe(1:Ns),:)*self.getfilter()'; 
            utemp=Upred.mu';    
            [z,c]=kmeans(utemp,NU);
            for i=1:NU
                NA = sum(z==i);
                modeltemp.u{i}.updateSS(self.invSigmau*mean(utemp(z==i,:))',self.invSigmau,NA);                    

                SEu1u1gz = [utemp(z==i,:)'*utemp(z==i,:),sum(utemp(z==i,:),1)';sum(utemp(z==i,:),1),NA];
                SEYu1gz = [Y(idxe(z==i),:)'*utemp(z==i,:),sum(Y(idxe(z==i),:),1)'];
                SEYYgz = Y(idxe(z==i),:)'*Y(idxe(z==i),:);
                self.AB{i}.updateSS(SEu1u1gz/NA,SEYu1gz/NA,SEYYgz/NA,NA);

            end                
            for i=1:NU
                self.AB{idx(i)}=modeltemp.AB{i};
                self.u{idx(i)}=modeltemp.u{i};
            end
            self.pi.alpha=self.pi.alpha_0;

        end
                       
        function MSE = plot(self,Y,X,fignum)
            
%            self.updateassignments(Y,X);
%  To really compute this correctly we need to set the assigments without
%  using Y.
            
            [Ypred,Upred]=self.getPredictions(X);
            MSE = mean((Y-Ypred.mu').^2);            
            
            if(self.NC==1)
                idx=ones(1,size(X,1));
            else
                [m,idx]=max(Upred.p);
            end
            
            cc=jet(self.NC);
            for i=1:self.NO
                figure(fignum+i-1)
                if(self.D==1)
                    scatter(Upred.mu(1,:),Y(:,i)',3*ones(size(Ypred.mu(i,:))),'k.')            
                    hold on
                    scatter(Upred.mu(1,:),Ypred.mu(i,:),9*ones(size(Ypred.mu(i,:))),cc(idx,:),'.')
                    hold off
                else
                    scatter3(Upred.mu(1,:),Upred.mu(2,:),Y(:,i)',9*ones(size(Ypred.mu(i,:))),'k.')            
                    hold on
                    scatter3(Upred.mu(1,:),Upred.mu(2,:),Ypred.mu(i,:),9*ones(size(Ypred.mu(i,:))),cc(idx,:),'.')
                    hold off
                end
            end
            if(self.D>1)
                figstart=fignum+self.NO;
                np=3;
                mu1 = mean(Upred.mu(1,:));
                std1 = std(Upred.mu(1,:));
                u1=linspace(mu1-2*std1,mu1+2*std1,np^2+1);

                [temp,idx2]=histc(Upred.mu(1,:),u1);

                for i=1:self.NO
                    figure(figstart+i-1)
                    k=0;
                    for j=1:np
                    for jj=1:np
                        k=k+1;
                        idx3=find(idx2==k);  
                        if(length(idx3)>0)
                           subplot(np,np,k), scatter(Upred.mu(2,idx3),Y(idx3,i)','k.');
                           hold on; scatter(Upred.mu(2,idx3),Ypred.mu(i,idx3),10*ones(1,length(idx3)),cc(idx(idx3),:),'.')
                           hold off;
                        end
                    end
                    end
                end
            end
        end
        
        function rotateW(self,X,Y)
           % This variation uses canonical correlation analysis to
           % determine the best rotation and scaling on U that predicts Y
           % (linearlly) and rotates U and W and AB to match.  
           
           Upreds = nan(size(X, 1), self.D);
           
           [Ypred,Upred]=self.getPredictions(X);
           [A,B]=canoncorr(Ypred.mu',Upred.mu'); %B is DxNO and we need DxD
           
           if(self.NO<self.D)
               B=[B,null(B')];
           else
               B=B(:,1:self.D);
           end
           
           for i=1:self.D
               B(:,i)=B(:,i)/norm(B(:,i));
           end
           P=B;
           
           colors = jet(self.NC);
           self.rotateU(P);
           
           [~, clust_label] = max(Upred.p', [], 2);
           
           [sp_nr, sp_nc] = BestArrayDims(self.D);
           figure; subplot(sp_nr, sp_nc, 1);
           for i = 1:self.NC
               scatter(Upred.mu(1, clust_label == i), Ypred.mu(clust_label == i), 50, colors(i, :), '.')
               hold on;
           end
           scatter(Upred.mu(1, :), Y, 20, [0 0 0], '.')
           xlabel('u_1')
           ylabel('y')
           Upreds(:, 1) = Upred.mu(1, :)';
           
           for i = 2:self.D
               us = nan(self.D, self.NC);
               for j = 1:self.NC
                   us(:, j) = self.u{j}.mu;
               end
               u_design = cell(self.D, 1);
               for j = 1:self.D
                   u_design{j} = repmat(us(:, j)', floor(self.NA(j)), 1);
               end
               u_design = cat(1, u_design{:});
               
               [B] = pca(u_design(:, i:end));
               
               %%%%% Find orthogonal dimensions %%%%%
               B = B(:, 1);
               B=[B,null(B')];
%                if(self.NO<self.D-i+1)
%                    B=[B,null(B')];
%                else
%                    %%%% ???? %%%%
%                    B=B(:,1:self.D);
%                end
               for j=1:self.D-i+1
                   B(:,j)=B(:,j)/norm(B(:,j));
               end
               %%%%%
               
               P = eye(self.D);
               P(i:end, i:end) = B;
               self.rotateU(P);
               
               [Ypred,Upred] = self.getPredictions(X);
               subplot(sp_nr, sp_nc, i)
               for k = 1:self.NC
                   scatter(Upred.mu(i, clust_label == k), Ypred.mu(clust_label == k), 50, colors(k, :), '.')
                   hold on;
               end
               scatter(Upred.mu(i, :), Y, 20, [0 0 0], '.')
               xlabel(sprintf('u_%d', i))
               ylabel('y')
               Upreds(:, i) = Upred.mu(i, :)';
           end
           
           figure;
           [sp_nr, sp_nc] = BestArrayDims(self.D.^2);
           for i = 1:self.D
               for j = i:self.D
                   subplot(sp_nr, sp_nc, (i-1)*self.D + j)
                   for k = 1:self.NC
                       scatter(Upreds(clust_label == k, i), Upreds(clust_label == k, j), 50, colors(k, :), '.')
                       hold on
                   end
                   xlabel(sprintf('u_%d', i))
                   ylabel(sprintf('u_%d', j))
               end
           end
        end
        
        function canoncorrU(self, X)
            [Y,U]=self.getPredictions(X);
            used_clusters = find(self.NA > 0.05*size(X, 1)); % at least 5% of the data must lie within the cluster
            clust_var = nan(self.NC, 1);
            for i = 1:length(used_clusters)
                effective_num_datapts = sum(U.p(used_clusters(i), :));
                clust_var(used_clusters(i)) = ...
                    sum(U.p(used_clusters(i), :).*Y.mu.^2)./effective_num_datapts - ...
                    (sum(U.p(used_clusters(i), :).*Y.mu)./effective_num_datapts).^2;
            end
            [~, idx] = min(clust_var);
            ylogistic = U.p(idx, :) > 0.75;
            theta = glmfit(U.mu', ylogistic', 'binomial'); % note that theta(1) is the intercept
            
%             if self.D == 3
%                 figure; scatter3(U.mu(1, :), U.mu(2, :), U.mu(3, :), 100, ylogistic)
%                 [x, y] = meshgrid(-4:0.1:2);
%                 z = -1/theta(4)*(theta(2)*x + theta(3)*y + theta(1));
%                 hold on; surf(x, y, z, zeros(size(z, 1)))
%             elseif self.D == 2
%                 figure; scatter(U.mu(1, :), U.mu(2, :), 100, ylogistic)
%                 x = -4:0.1:2;
%                 y = -1/theta(3)*(theta(2)*x + theta(1));
%                 hold on; plot(x, y, 'k-')
%             end
            
            if self.NO < self.D
                B = [theta(2:self.D+1),null(theta(2:self.D+1)')];
                for i = 1:self.D
                    B(:,i) = B(:,i)/norm(B(:,i));
                end
                if(self.D>2)
                    U.mu=B'*U.mu;
                    [~,Bcc]=canoncorr(Y.mu',U.mu(2:end, :)'); %B is DxNO and we need DxD
                    Bcc = Bcc/norm(Bcc);
                    Bcc=[Bcc,null(Bcc')];
                    V(2:self.D,2:self.D)=Bcc;
                    V(1,1)=1;
                    B=B*V;
                end
            else
                B=eye(self.D);
            end
            
            B=B';  % cononcorr assumes B is a right operator on a data matrix
            % so B' is a left operator on vector u;
            
            self.rotateU(B);
            
            [Y,U]=self.getPredictions(X);
            Bcorrection = 1;
            if self.D == 1
                if corr(Y.mu', U.mu') < 0
                    Bcorrection = -1;
                end
            end
            if self.D >= 2
                Bcorrection = eye(self.D);
                if mean(U.mu(1, ylogistic)) > mean(U.mu(1, ~ylogistic))
                    Bcorrection(1,1) = -1;
                end
                if corr(Y.mu', U.mu(2, :)') < 0
                    Bcorrection(2,2) = -1;
                end
            end
            if self.D == 3
                Bcorrection(3,3) = 1;
            end
            
            Bcorrection=Bcorrection';
            self.rotateU(Bcorrection);
        end
        
        function canoncorrU2(self,X)
            
           [Y,U]=self.getPredictions(X);
           [A,B]=canoncorr(Y.mu',U.mu'); %B is DxNO and we need DxD
           
           if(self.NO<self.D)
               B=[B,null(B')];
               for i=1:self.D
                   B(:,i)=B(:,i)/norm(B(:,i));
               end   
               if(self.D>2)
                   U.mu=U.mu'*B;
                   V(2:self.D,2:self.D)=pca(U.mu(:,2:self.D));
                   V(1,1)=1;
                   B=B*V;
               end
           else
               B=B(:,1:self.D);
           end
           
           B=B';  % cononcorr assumes B is a right operator on a data matrix
                  % so B' is a left operator on vector u;
                  
           self.rotateU(B);
        end
        
        function rotateU(self,P)
           % Takes in a rotation matrix P that left operates on vector U           
                  
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
        
        function [Ypred,Upred] = getPredictions(self,X)
            
            Ns=size(X,1);
            logptilde = repmat(self.pi.loggeomean,1,Ns);

            for k=1:self.NC

                abar = self.AB{k}.mean;
                bbar = abar(:,self.D+1);
                abar = abar(:,1:self.D);
                abm2 = self.AB{k}.EXTinvUX;

                ugz{k}.invSigma = self.W.EXTinvUX + self.invSigmau; 
                ugz{k}.Sigma = inv(ugz{k}.invSigma);

                ugz{k}.invSigmamu = (X*self.W.EXTinvU')';
                ugz{k}.invSigmamu = bsxfun(@plus,ugz{k}.invSigmamu,self.invSigmau*self.u{k}.mean);
                ugz{k}.mu=ugz{k}.Sigma*ugz{k}.invSigmamu;   

                yugz{k}.invSigma = [self.AB{k}.EinvU,-self.AB{k}.EinvU*abar;...
                                -abar'*self.AB{k}.EinvU,abm2(1:self.D,1:self.D) + ugz{k}.invSigma];
                yugz{k}.Sigma = inv(yugz{k}.invSigma);
                
                yugz{k}.invSigmamu = repmat([self.AB{k}.EinvU*bbar;-abm2(1:self.D,self.D+1);],1,Ns);
                yugz{k}.invSigmamu(self.NO+1:end,:) = yugz{k}.invSigmamu(self.NO+1:end,:) ...
                                                    + ugz{k}.invSigmamu;
                yugz{k}.mu = yugz{k}.Sigma*yugz{k}.invSigmamu;
                
                ygz{k}.mu = yugz{k}.mu(1:self.NO,:);
                ygz{k}.Sigma = yugz{k}.Sigma(1:self.NO,1:self.NO);
                
                logptilde(k,:) = logptilde(k,:) + 1/2*sum(ugz{k}.mu.*ugz{k}.invSigmamu,1) ...
                    + 1/2*log(det(ugz{k}.Sigma)) + self.D/2*log(2*pi);

                %add everything that has no ugz in it.
                logptilde(k,:) = logptilde(k,:) ...
                    - 1/2*sum(sum(self.u{k}.secondmoment.*self.invSigmau)) - self.D/2*log(2*pi) ...
                    + 1/2*self.logdetinvSigmau;
                
                yuinvZ(k,1)=1/sqrt(det(yugz{k}.Sigma));
            end
            
            if(self.NC==1);
               p=ones(1,Ns);
               self.NA(1)=Ns;
               logptilde=zeros(1,Ns); 
            else
                p = exp(bsxfun(@minus,logptilde,max(logptilde)));
                p = bsxfun(@rdivide,p,sum(p,1));
                self.NA = sum(p,2);
            end %if
% This code computes mean and variance   
            Upred.p = p;            
            Ypred.mu = zeros(self.NO,Ns);
            Upred.mu = zeros(self.D,Ns);
%             Ypred.Sigma = zeros(self.NO,Ns);
            Ypred.Sigma = zeros(self.NO^2,Ns);
            for k=1:self.NC
                Upred.mu = Upred.mu + bsxfun(@times,ugz{k}.mu,p(k,:));
                Ypred.mu = Ypred.mu + bsxfun(@times,ygz{k}.mu,p(k,:));                
%                Ypred.Sigma = Ypred.Sigma + diag(ygz{k}.Sigma)*p(k,:);                
                Ypred.Sigma = Ypred.Sigma + ygz{k}.Sigma(:)*p(k,:);                                
            end
            Ypred.Sigma = reshape(Ypred.Sigma,self.NO,self.NO,Ns);

% % This code approximates the Maximum likelihood estimate of the joint
% % distribution on U and Y.
%             [m,loc]=max(bsxfun(@times,p,yuinvZ));
%             p=zeros(self.NC,Ns);
%             p([0:self.NC:(Ns-1)*self.NC]+loc)=1;
%             for k=1:self.NC
%                 Upred.mu = Upred.mu + bsxfun(@times,ugz{k}.mu,p(k,:));
%                 Ypred.mu = Ypred.mu + bsxfun(@times,ygz{k}.mu,p(k,:));                
%             end
            

        end
 
        function KL = KLqprior(self)
            KL = self.pi.KLqprior
            KL = KL + self.W.KLqprior;
            for k=1:self.NC
                KL = KL + self.AB{k}.KLqprior;
                KL = KL + self.u{k}.KLqprior;
            end
        end
                
        function [Xpred] = Decode(self,Y,Xprior)
            
            if(~exist('Xprior','var'))
                Xprior.invSigma=zeros(self.NR);
                Xprior.invSigmamu=zeros(self.NR,1);
            end
            % Assumes that Y is Ns x NO and X is Ns x NR
            
            % In order to compute the cluster assignments we first compute
            % the approximate posterior over u given z 
            % Fortunately under the variational approximation all u(t,k)
            % have the same precision matrix for a given k.  Meanwhile, 
            % because of the conditional approximation, cluster assignments
            % only depende upon the u undependent terms, and the partition
            % function of the posterior on the u{k}'s.
            DL=self.L;
            Ns=size(Y,1);
            logptilde = repmat(self.pi.loggeomean,1,Ns);

            for k=1:self.NC
                abar = self.AB{k}.mean;
                bbar = abar(:,self.D+1);
                abar = abar(:,1:self.D);
                yminusb = bsxfun(@minus,Y',bbar);

                ugz{k}.invSigma = abar'*self.AB{k}.invU.mean*abar ...
                    + self.NO*self.AB{k}.V(1:self.D,1:self.D) ...
                    + self.invSigmau; 

                ugz{k}.Sigma = inv(ugz{k}.invSigma);

                ugz{k}.invSigmamu = abar(:,1:self.D)'*self.AB{k}.invU.mean*yminusb;
                ugz{k}.invSigmamu = bsxfun(@minus,ugz{k}.invSigmamu,self.NO*self.AB{k}.V(1:self.D,self.D+1));
                ugz{k}.invSigmamu = bsxfun(@plus,ugz{k}.invSigmamu,self.invSigmau*self.u{k}.mean);
                ugz{k}.mu=ugz{k}.Sigma*ugz{k}.invSigmamu;   

% big matrix inversions here.... look for fix                  
                uxgz.invSigma = [ugz{k}.invSigma+self.W.EXTinvUX,-self.W.EXTinvU;...
                                    -self.W.EXTinvU',self.W.EinvU+Xprior.invSigma];
                uxgz.invSigmamu = [ugz{k}.invSigmamu;repmat(Xprior.invSigmamu,1,Ns);];
              
                uxgz.Sigma = inv(uxgz.invSigma);
                uxgz.mu = uxgz.Sigma*uxgz.invSigmamu;
                
                xgz{k}.Sigma = uxgz.Sigma(self.D+1:end,self.D+1:end);
                xgz{k}.mu = uxgz.mu(self.D+1:end,:);
                xgz{k}.invSigma = inv(xgz{k}.Sigma);
                xgz{k}.invSigmamu = xgz{k}.invSigma*xgz{k}.mu;

                xinvZ(k,1)=1/sqrt(det(xgz{k}.Sigma));
                
                logptilde(k,:) = logptilde(k,:) + 1/2*sum(ugz{k}.mu.*ugz{k}.invSigmamu,1) ...
                    + 1/2*log(det(ugz{k}.Sigma)) + self.D/2*log(2*pi);

                %add everything that has no ugz in it.
                logptilde(k,:) = logptilde(k,:) ...
                    - 1/2*sum(self.AB{k}.invU.mean*(yminusb).*yminusb,1) ...
                    - 1/2*self.NO*self.AB{k}.V(self.D+1,self.D+1) ...
                    + 1/2*self.AB{k}.invU.Elogdet - self.NO/2*log(2*pi) ...
                    - 1/2*trace(self.u{k}.secondmoment*self.invSigmau) - self.D/2*log(2*pi) ...
                    + 1/2*self.logdetinvSigmau;
                
            end
                % Ok so this gives logptilde(:,k) which must be converted
                % to an actual probability.
            if(self.NC==1);
               p=ones(1,Ns);
            elseif(~exist('p','var'))
                p = exp(bsxfun(@minus,logptilde,max(logptilde)));
                p = bsxfun(@rdivide,p,sum(p,1));
            end %if            
            
% At this point we have computed p(u|z,y), p(z|y), and p(x|z,y)
% This code computes the mean of X       
            Xpred.mu=zeros(self.NR,Ns);
            for k=1:self.NC
                Xpred.mu = Xpred.mu + bsxfun(@times,xgz{k}.mu,p(k,:));
            end
            Xpred.mu=Xpred.mu';

% This code approximates the Maximum likelihood estimate of X
%
%             [m,loc]=max(bsxfun(@times,p,xinvZ));
%             p=zeros(self.NC,Ns);
%             p([0:self.NC:self.NC*(Ns-1)]+loc)=1;
%             Xpred.mu=zeros(self.NR,Ns);
%             for k=1:self.NC
%                 Xpred.mu = Xpred.mu + bsxfun(@times,xgz{k}.mu,p(k,:));                
%             end            
            
            
        end

    end
   
end
