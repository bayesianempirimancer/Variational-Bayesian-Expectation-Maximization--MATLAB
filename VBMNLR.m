classdef VBMNLR < handle
    
    % performs variational bayesian logistic regression.  Bouchard 2008 but with 
    % access to a more complex set of priors on beta{k}.  
    % Here X(t,:) is a set of regressors and Y(t) is a set of class labels k=1:K
    %      for data point t
    
    % Usage
    % Make sure Y is intergers >=1
    % Augment regressors to include a 1
    %   X=[X,ones(size(X,1),1)];
    %
    % Define model and fit
    %   sparse=0;
    %   model=VBMNLR(max(Y),size(X,2),sparse)
    %   DL=model.update(Y,X,200) % convergence occurs when DL is small
    %                            % warnings about negative DL are ok 
    %                            % provided DL/model.L is very small 
    %                            % and it alwyas is.
    % Get predictions (note that Yhat is class labels and p is posterior
    %                   probability of assignment)
    %   [Yhat,p]=model.getPredictions(X,10);
    %
    %   If the number of data points is very large you can use 
    %       [Yhat,p]=model.getSimplePredictions(X);
    %   to speed things up.  
    %
    % Compute percent correct and confusion matrix in the usual way
    %   pc=mean(Yhat==Y);
    %   for k=1:max(Y)
    %   for j=1:max(Y)
    %       confusion(k,j) = sum(Yhat==k & Y==j)/sum(Y==j);
    %   end
    %   end
    %   
    %   The variable P can be used to compute mutual information
    %
    %   MI = - sum(mean(p).*log2(mean(p)) + mean(sum(p.*log2(p),2));
    %
            
    
    properties
        K %number of classes
        NR %number of regressors        

        beta % beta{k}=MVN(NR,1) or normalsparse(NR)
        xi % xi(k) is the variational parameter for class k vector beta
        lambda
        alpha % variational parameter for softmax
        
        SEXX
        SEYX % 
        n
        
        L
        iters
    end
    
    methods
        function self = VBMNLR(K,NR,issparse,priorprecision)  %is sparse = 0 if not sparse
                                           %          = Expected number of active
                                           %            regressors otherwise
            if(~exist('priorprecision','var'))
                priorprecision=1;
            end
            if(~exist('issparse','var'))
                issparse=0;
            end            
            self.NR=NR;
            self.K=K;
            if(~exist('issparse','var'))
                issparse = 0;
            end
            if(issparse==0)
                for k=1:K
                    self.beta{k} = dists.expfam.MVN(zeros(NR,1),priorprecision*eye(NR)); %second arg is precision of prior
                end
            else
                for k=1:K
                    self.beta{k} = dists.normalsparse(NR,issparse/NR,issparse/NR);
                end
            end
            self.iters=0;
            self.L=-Inf;            
            self.alpha=(self.K/2-1)/2;            
 
        end
        
        function fit(self,Y,X,maxiters,tol)
            DL = self.update(Y,X,1);
            if(~exist('maxiters','var'))
                maxiters=100;
            end
            if(~exist('tol','var'))
                tol=1e-12*size(Y,1);
            end
            
            while((DL/abs(self.L)>1e-12 | self.iters<5) & self.iters < maxiters)
                DL = self.update(Y,X,1);
                plot(self.iters,DL/abs(self.L),'o')
                hold on
            end
            if(self.iters == maxiters) 
                fprintf('maxiters reached\n')
            end
            hold off
        end
        
        function DL = update(self,Y,X,niters)
            if(~exist('niters','var'))
                niters = 1;
            end
            for i=1:niters
                DL = self.update_suffstats(Y,X);
                self.updateparms;
                self.iters=self.iters+1;
                if(DL<0)
                fprintf(['Warning Lower Bound Decreasing (DL/L = ',num2str(DL/abs(self.L)),') at iteration ',num2str(self.iters),'\n'])
                end
            end
        end
        
        function DL = update_suffstats(self,Y,X,p)
            
            DL=self.L;
            Ns=size(X,1);
            if(~exist('p','var'))
                p=ones(Ns,1);
                self.n = Ns;
            else
                self.n=sum(p);
            end
            for k=1:self.K            
                self.xi(:,k) = sum((X*self.beta{k}.secondmoment).*X,2) ...
                             + self.alpha.^2 - 2*self.alpha.*(X*self.beta{k}.mean);                         
            end
            self.xi = sqrt(self.xi);
            self.lambda = 0.5./self.xi.*(1./(1+exp(-self.xi))-0.5);
            
            self.alpha=(self.K/2-1)/2;            
            for k=1:self.K
                self.alpha = self.alpha + self.lambda(:,k).*(X*self.beta{k}.mean);
            end
            self.alpha = self.alpha./(sum(self.lambda,2));
            
            for k=1:self.K            
                self.xi(:,k) = sum((X*self.beta{k}.secondmoment).*X,2) ...
                             + self.alpha.^2 - 2*self.alpha.*(X*self.beta{k}.mean);                         
            end
            self.xi = sqrt(self.xi);
            self.lambda = 0.5./self.xi.*(1./(1+exp(-self.xi))-0.5);

            for k=1:self.K 
                self.SEXX{k} = 2*X'*bsxfun(@times,self.lambda(:,k).*p,X);
                self.SEYX(k,:)  = ((real(Y==k) - 1/2 + 2*self.alpha.*self.lambda(:,k)).*p)'*X;
            end
            
            c = (self.K/2-1)*sum(self.alpha) + 1/2*sum(self.xi(:)) ...
              - sum(sum(self.lambda.*(bsxfun(@plus,self.alpha.^2,-self.xi.^2)))) ...
              - sum(sum(log(1+exp(self.xi))));
            
            self.L =  Ns*self.K*self.NR/2 - self.KLqprior + c;
            for k=1:self.K
                self.L = self.L + self.SEYX(k,:)*self.beta{k}.mean - 1/2*sum(sum(self.beta{k}.secondmoment.*self.SEXX{k}));
            end

            DL=self.L-DL;

        end

        function updateparms(self)
            for k=1:self.K
                self.beta{k}.updateSS(self.SEYX(k,:)'/self.n,self.SEXX{k}/self.n,self.n);
            end
        end
        
        function [Yhat,p] = getPredictions(self,X,iters)  %not vectorized and so very slow
            %%%%% NEED FIXING!  Go back to basics...
            tic
            if(~exist('iters','var'))
                iters=10;
            end
            Ns=size(X,1);
            xi=zeros(1,self.K);
            alpha=0;

            for k=1:self.K
                invSigma_0{k} = self.beta{k}.invSigma;
                Sigma_0{k}=inv(invSigma_0{k});
                invSigmamu_0(:,k) = self.beta{k}.invSigmamu;
                mu_0(:,k)=self.beta{k}.mean;
                invSigma{k} = self.beta{k}.invSigma;
                Sigma{k}=inv(invSigma{k});
                invSigmamu(:,k) = self.beta{k}.invSigmamu;
                mu(:,k)=self.beta{k}.mean;
            end
            
            p=zeros(Ns,self.K);
            logp=zeros(Ns,self.K);
            
            for n=1:Ns
                for k=1:self.K
                    mu=mu_0;
                    invSigma=invSigma_0;
                    Sigma=Sigma_0;
                    invSigmamu=invSigmamu_0;
                    alpha=0;
                    for i=1:iters
                        for kk=1:self.K
                            xi(kk) = X(n,:)*(Sigma{k}+mu(:,kk)*mu(:,kk)')*X(n,:)' ...
                                   + alpha^2 - 2*alpha*X(n,:)*mu(:,kk);
                        end
                        xi=sqrt(xi);
                        lambda = 0.5./xi.*(1./(1+exp(-xi))-0.5);
         
                        alpha=(self.K/2-1)/2;            
                        for kk=1:self.K
                            alpha = alpha + lambda(kk).*(X(n,:)*mu(:,kk));
                        end
                        alpha = alpha/(sum(lambda));

                        for kk=1:self.K
                            invSigma{kk} = invSigma_0{kk} + 2*lambda(kk)*X(n,:)'*X(n,:);
                            Sigma{kk}=inv(invSigma{kk});
                            invSigmamu(:,kk) = invSigmamu(:,kk) + (real(kk==k)-1/2+2*alpha*lambda(kk))*X(n,:)';
                            mu(:,kk) = inv(invSigma{kk})*invSigmamu(:,kk);
                        end
                    end
                    %lowerbound
                    logp(n,k)=(1-1/2+2*alpha*lambda(k))*X(n,:)*mu(:,k);
                end
                logp(n,:)=logp(n,:)-max(logp(n,:));
                p(n,:)=exp(logp(n,:))/sum(exp(logp(n,:)));
                if(mod(100*n/Ns,5)==0)
%                    ['precent complete = ',num2str(n/Ns*100), ' in ',num2str(toc/60),' minutes']
                    [num2str(toc/n*(Ns-n)),' seconds to completion']
                end
            end
            [m,Yhat]=max(logp');
            Yhat=Yhat';
            
        end

        function [Yhat,p] = getSimplePredictions(self,X)
            for k=1:self.K
                logp(:,k) = X*self.beta{k}.mean;
            end
            logp=bsxfun(@plus,logp,-max(logp')');
            p=exp(logp)./sum(exp(logp),2);
            [m,Yhat]=max(logp');
            Yhat=Yhat';
        end
        
        function KL = KLqprior(self)
            KL=0;
            for k=1:self.K
                KL = KL + self.beta{k}.KLqprior;
            end
        end
        
        function res = betamat(self)
            res=zeros(self.K,self.NR);
            for k=1:self.K
                res(k,:)=self.beta{k}.mean()';
            end            
        end
        
    end
end
