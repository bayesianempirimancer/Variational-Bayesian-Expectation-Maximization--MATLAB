classdef VBLR < handle
    
    % performs variational bayesian logistic regression.  As in Jan D.'s paper with 
    % one exception.  X is a set of regressors and Y is binary
    properties
        
        NR %number of regressors

        beta
        xi
        SEXX
        SEYX
        n
        
        L
        iters
    end
    
    methods
        function self = VBLR(NR,issparse)  %is sparse = 0 if not sparse
                                           %          = Expected number of active
                                           %            regressors otherwise
            self.NR=NR;
            if(~exist('issparse','var'))
                issparse = 0;
            end
            if(issparse==0)
                self.beta = dists.expfam.MVN(zeros(NR,1),eye(NR)); %second arg is precision of prior
            else
                self.beta = dists.normalsparse(NR,issparse/NR,issparse/NR);
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
            
            self.xi = sqrt(sum((X*self.beta.secondmoment).*X,2));
            self.SEXX = X'*bsxfun(@times,X,self.lambda(self.xi).*p)*2;
            self.SEYX = (((Y-0.5).*p)'*X);  

            
            self.L = self.SEYX*self.beta.mean + sum(self.lnsigma(self.xi)) - sum(self.xi)/2 ...
                   - self.KLqprior;
            DL=self.L-DL;

        end

        function updateparms(self)
            self.beta.updateSS(self.SEYX'/self.n,self.SEXX/self.n,self.n);
        end
        
        function [Yhat,iters] = getPredictions(self,X,tol)
            xiplus = sqrt(sum((X*self.beta.secondmoment).*X,2));
            ximinus = sqrt(sum((X*self.beta.secondmoment).*X,2));
            
            XTSigma = (X*self.beta.ESigma);
            XTSigmaX = sum(XTSigma.*X,2);
            if(~exist('tol','var'))
                tol=1e-8;
            end
            logodds = Inf(size(xiplus));
            logoddsold = zeros(size(xiplus));
            iters=0;
            while(max(abs(logodds-logoddsold))>tol)
                iters=iters+1;
                logoddsold = logodds;
                denom = 0.5./self.lambda(xiplus) + XTSigmaX;
                bplus = bsxfun(@times,XTSigma,(0.5-(X*self.beta.mean+0.5*XTSigmaX)./denom));
                bplus = bsxfun(@plus,bplus,self.beta.mean');
                xiplus = sqrt(XTSigmaX - XTSigmaX.^2./denom + sum(bplus.*X,2).^2);

                denom = 0.5./self.lambda(ximinus) + XTSigmaX;
                bminus = bsxfun(@times,XTSigma,(-0.5-(X*self.beta.mean-0.5*XTSigmaX)./denom));
                bminus = bsxfun(@plus,bminus,self.beta.mean');
                ximinus = sqrt(XTSigmaX - XTSigmaX.^2./denom + sum(bminus.*X,2).^2);
                
                logodds = 0.5*log(1+2*self.lambda(xiplus).*XTSigmaX) ...
                        + sum(bplus.*bsxfun(@plus,(self.beta.ESigma*self.beta.mean)',0.5*X),2) ...
                        + self.lnsigma(xiplus) - 0.5*xiplus + self.lambda(xiplus).*xiplus.^2 ...
                        - 0.5*log(1+2*self.lambda(ximinus).*XTSigmaX) ...
                        - sum(bminus.*bsxfun(@plus,(self.beta.ESigma*self.beta.mean)',-0.5*X),2) ...
                        - self.lnsigma(ximinus) + 0.5*ximinus - self.lambda(ximinus).*ximinus.^2;         
                    
            end
            
            Yhat = 1./(1+exp(-logodds));
        end
                
        function res = lnsigma(self,xi)
            res = -log(1+exp(-xi));
        end
        
        function res = lambda(self,xi)
            res = 0.5./xi.*(1./(1+exp(-xi))-0.5);
        end
         
        function KL = KLqprior(self)
            KL = self.beta.KLqprior;
       end
                
    end
   
end
