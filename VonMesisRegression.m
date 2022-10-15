classdef VonMesisRegression < handle
    
    % performs von-mises regression.  X is a set of regressors and Y is a 
    % angular variable in radians with period 2*pi.
    properties
        
        NR %number of regressors
        a
        b
        xi
        SEXX
        SEcosYX
        SEsinYX
        n
        
        L
        iters
    end
    
    methods
        function self = VonMesisRegression(NR,issparse)
            self.NR=NR;
            if(~exist('issparse','var'))
                issparse = 0;
            end
            if(issparse==0)
                self.a = dists.expfam.MVN(zeros(NR,1),eye(NR)); %second arg is precision of prior
                self.b = dists.expfam.MVN(zeros(NR,1),eye(NR));
            else
                self.a = dists.normalsparse(NR,issparse/NR,issparse/NR);
                self.b = dists.normalsparse(NR,issparse/NR,issparse/NR);
            end
            self.iters=0;
            self.L=-Inf;
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
            Ns=size(Y,1);
            if(~exist('p','var'))
                p=ones(Ns,1);
                self.n = Ns;
            else
                self.n=sum(p);
            end
            
            self.xi = sum(X*(self.a.secondmoment+self.b.secondmoment).*X,2);
            self.SEXX = X'*bsxfun(@times,X,self.fprime(self.xi).*p)*2;
            self.SEcosYX = ((cos(Y).*p)'*X);
            self.SEsinYX = ((sin(Y).*p)'*X);

            
            self.L = self.SEcosYX*self.a.mu + self.SEsinYX*self.b.mu - sum(self.f(self.xi)) ...
                   - self.KLqprior;
            DL=self.L-DL;

        end

        function updateparms(self)
            self.a.updateSS(self.SEcosYX'/self.n,self.SEXX/self.n,self.n);
            self.b.updateSS(self.SEsinYX'/self.n,self.SEXX/self.n,self.n);
        end
        
        function res = f(self,xi)
            res = log(besseli(0, sqrt(xi),1))+sqrt(xi);
        end
        
        function res = fprime(self,xi)
            res = besseli(1,sqrt(xi),1)./besseli(0,sqrt(xi),1)./sqrt(xi)/2;
        end
        
        function [mu,kappa] = getPredictions(self,X)
            
            mu = angle(X*self.a.mean + sqrt(-1)*X*self.b.mean);
            kappa = sqrt(sum((X*(self.a.secondmoment+self.b.secondmoment)).*X,2));
            
        end
        
        function res = getPdf2(self,theta,X)
              
            [mu,kappa]=self.getPredictions(X);
            
            for n=1:size(X,1)
                
                [m,j]=min(abs(mu(n,1)-theta));
                xi=(kappa(n,1))^2;
                xi=mean(self.xi)
                xiold=Inf;
                Pa = self.a.EinvSigma + 2*self.fprime(xi)*X(n,:)'*X(n,:);
                Pb = self.b.EinvSigma + 2*self.fprime(xi)*X(n,:)'*X(n,:);
                
                invPa=inv(Pa);
                invPb=inv(Pb);
                
                mua = invPa*(self.a.EinvSigma*self.a.mean + X(n,:)'*cos(theta(j)));
                mub = invPb*(self.b.EinvSigma*self.b.mean + X(n,:)'*sin(theta(j)));
                xi = X(n,:)*(invPa+invPb)*X(n,:)' + (X(n,:)*mua)^2 + (X(n,:)*mub)^2;
                iter=0;
                while(abs(xi-xiold)>0.0000001)
                    iter=iter+1;
                    xiold=xi;
                    Pa = self.a.EinvSigma + 2*self.fprime(xi)*X(n,:)'*X(n,:);
                    Pb = self.b.EinvSigma + 2*self.fprime(xi)*X(n,:)'*X(n,:);

                    invPa=inv(Pa);
                    invPb=inv(Pb);

                    for j=1:length(theta)
                        
                        mua = invPa*(self.a.EinvSigma*self.a.mean + X(n,:)'*cos(theta(j)));
                        mub = invPb*(self.b.EinvSigma*self.b.mean + X(n,:)'*sin(theta(j)));

                        logp1(j) = -1/2*self.a.mean'*self.a.EinvSigma*self.a.mean + 1/2*mua'*invPa*mua ...
                           -1/2*log(det(Pa)) + 1/2*self.a.ElogdetinvSigma ...
                           -1/2*self.b.mean'*self.b.EinvSigma*self.b.mean + 1/2*mub'*invPb*mub ...
                           -1/2*log(det(Pb)) + 1/2*self.b.ElogdetinvSigma ... 
                           - self.f(xi) + self.fprime(xi)*(xi);
                    end
                    [m,j]=max(logp1);
                    mua = invPa*(self.a.EinvSigma*self.a.mean + X(n,:)'*cos(theta(j)));
                    mub = invPb*(self.b.EinvSigma*self.b.mean + X(n,:)'*sin(theta(j)));


                    xi = X(n,:)*(invPa+invPb)*X(n,:)' + (X(n,:)*mua)^2 + (X(n,:)*mub)^2;
                    [n,j,iter,sqrt(xi)]
                end
                Pa = self.a.EinvSigma + 2*self.fprime(xi)*X(n,:)'*X(n,:);
                Pb = self.b.EinvSigma + 2*self.fprime(xi)*X(n,:)'*X(n,:);

                invPa=inv(Pa);
                invPb=inv(Pb);
                for j=1:length(theta)
                    mua = invPa*(self.a.EinvSigma*self.a.mean + X(n,:)'*cos(theta(j)));
                    mub = invPb*(self.b.EinvSigma*self.b.mean + X(n,:)'*sin(theta(j)));
                    logp(j,n) = -1/2*self.a.mean'*self.a.EinvSigma*self.a.mean + 1/2*mua'*invPa*mua ...
                           -1/2*log(det(Pa)) + 1/2*self.a.ElogdetinvSigma ...
                           -1/2*self.b.mean'*self.b.EinvSigma*self.b.mean + 1/2*mub'*invPb*mub ...
                           -1/2*log(det(Pb)) + 1/2*self.b.ElogdetinvSigma ... 
                           - self.f(xi) + self.fprime(xi)*(xi);
                end
            end
            
           res = bsxfun(@plus,logp,-max(logp));
           res = bsxfun(@times,exp(res),1./sum(exp(res),1));
        end
        
        function res = getPdf(self,theta,X)
            [mu,kappa]=self.getPredictions(X);
            dtheta = ones(size(mu))*theta-mu*ones(size(theta));
            res = bsxfun(@times,exp(bsxfun(@times,cos(dtheta),kappa)),1./besseli(0,kappa))/2/pi;
        end
 
        function KL = KLqprior(self)
            KL = self.a.KLqprior + self.b.KLqprior;
       end
                
    end
   
end
